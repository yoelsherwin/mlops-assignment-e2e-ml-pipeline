import os
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow.decorators import dag, task
from airflow.models.param import Param  # available on Airflow 2.x and 3.x
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# get_current_context moved to the Task SDK in Airflow 3.x; fall back for 2.x.
try:
    from airflow.sdk import get_current_context  # Airflow 3.x
except ImportError:  # Airflow 2.x
    from airflow.operators.python import get_current_context

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"

# Image built from the project Dockerfile; the agent/eval steps run inside it.
PIPELINE_IMAGE = os.environ.get("PIPELINE_IMAGE", "mlops-pipeline:latest")
# Paths *inside* the agent/eval container (Dockerfile WORKDIR is /mlops-assignment).
CONTAINER_ROOT = "/mlops-assignment"
CONTAINER_RUNS = f"{CONTAINER_ROOT}/runs"
# Host path of runs/ for DockerOperator bind-mounts: in docker-compose Airflow runs in a
# container, so the mount SOURCE must be the host path (HOST_PROJECT_DIR); in standalone
# Airflow is on the host, so PROJECT_ROOT already is the host path.
HOST_PROJECT_DIR = Path(os.environ.get("HOST_PROJECT_DIR", str(PROJECT_ROOT)))
HOST_RUNS = HOST_PROJECT_DIR / "runs"

# Reliability defaults: retries absorb transient docker/network/API hiccups; the timeout
# guards the long agent/eval steps against hangs (raise it for large slices).
STEP_TIMEOUT = timedelta(hours=2)

# subset -> SWE-bench dataset name that the eval harness expects
DATASET_BY_SUBSET = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
}


def get_git_sha(project_root: Path) -> str:
    """Best-effort HEAD commit of the pipeline repo, for provenance."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def build_config(ctx) -> dict:
    """Freeze the Airflow params + provenance into the run's config record."""
    params = ctx["params"]

    run_id = params["run_id"] or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    subset = params["subset"]
    if subset not in DATASET_BY_SUBSET:
        raise ValueError(
            f"Unknown subset {subset!r}; expected one of {sorted(DATASET_BY_SUBSET)}"
        )

    return {
        "run_id": run_id,
        "airflow_run_id": ctx["run_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": params["split"],
        "subset": subset,
        "dataset_name": DATASET_BY_SUBSET[subset],
        "model": params["model"],
        "task_slice": params["task_slice"],
        "workers": params["workers"],
        "cost_limit": params["cost_limit"],
        "git_sha": get_git_sha(PROJECT_ROOT),
    }


def build_manifest(run_dir: Path) -> dict:
    """Index a finished run: metadata + relative pointers to every key artifact,
    so the folder is self-describing and portable (move/zip/upload and links hold).
    """
    config = json.loads((run_dir / "config.json").read_text())
    run_id = config["run_id"]

    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    run_agent = run_dir / "run-agent"
    run_eval = run_dir / "run-eval"
    preds = run_agent / "preds.json"
    trajectories = sorted(run_agent.rglob("*.traj.json"))
    eval_reports = sorted(run_eval.rglob(f"*.{run_id}.json"))
    eval_report = eval_reports[0] if eval_reports else None
    eval_logs = run_eval / "logs"

    def rel(p):
        return p.relative_to(run_dir).as_posix() if p and p.exists() else None

    mlflow_ref_path = run_dir / "mlflow_run.json"
    if mlflow_ref_path.exists():
        mlflow_ref = json.loads(mlflow_ref_path.read_text())
    else:
        mlflow_ref = {
            "tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            "experiment": os.environ.get("MLFLOW_EXPERIMENT_NAME", "swebench-eval"),
            "run_name": run_id,
        }

    return {
        "run_id": run_id,
        "created_at": config.get("created_at"),
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": config.get("git_sha"),
        "config": {
            k: config.get(k)
            for k in ["split", "subset", "dataset_name", "model", "task_slice", "workers", "cost_limit"]
        },
        "metrics": metrics,
        "artifacts": {
            "config": "config.json",
            "metrics": rel(metrics_path),
            "predictions": rel(preds),
            "trajectories": [t.relative_to(run_dir).as_posix() for t in trajectories],
            "eval_report": rel(eval_report),
            "eval_logs": rel(eval_logs),
        },
        "artifact_location": {
            "local": f"runs/{run_id}",
            "remote": None,  # set by the Phase 6 S3 upload
        },
        "mlflow": mlflow_ref,
        "inventory": {
            "num_predictions": len(json.loads(preds.read_text())) if preds.exists() else 0,
            "num_trajectories": len(trajectories),
        },
    }


def log_to_mlflow(run_dir: Path, config: dict, metrics: dict) -> None:
    """Log params, metrics, and key artifacts to MLflow in-process, and drop an
    mlflow_run.json in the run dir so manifest.json can link back to the tracked run.

    Uses the mlflow-skinny client present in the Airflow env, reaching
    MLFLOW_TRACKING_URI (localhost:5000 in standalone, http://mlflow:5000 in compose).
    """
    import mlflow  # lazy: only this step needs the client

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME", "swebench-eval")
    mlflow.set_experiment(experiment)

    run_id = config["run_id"]
    eval_reports = sorted((run_dir / "run-eval").rglob(f"*.{run_id}.json"))
    eval_report = eval_reports[0] if eval_reports else None

    param_keys = ["run_id", "airflow_run_id", "split", "subset", "dataset_name",
                  "model", "task_slice", "workers", "cost_limit", "git_sha"]
    metric_keys = ["total_instances", "submitted_instances", "completed_instances",
                   "resolved_instances", "unresolved_instances", "empty_patch_instances",
                   "error_instances", "resolve_rate"]

    with mlflow.start_run(run_name=run_id) as run:
        mlflow.log_params({k: config.get(k) for k in param_keys})
        mlflow.log_metrics({k: metrics[k] for k in metric_keys if k in metrics})
        mlflow.set_tag("artifact_dir", str(run_dir))
        for f in (run_dir / "config.json", run_dir / "metrics.json", eval_report):
            if f and Path(f).exists():
                mlflow.log_artifact(str(f))
        info = run.info

    tracking_uri = mlflow.get_tracking_uri()
    (run_dir / "mlflow_run.json").write_text(json.dumps({
        "tracking_uri": tracking_uri,
        "experiment": experiment,
        "experiment_id": info.experiment_id,
        "mlflow_run_id": info.run_id,
        "run_name": run_id,
        "run_url": f"{tracking_uri.rstrip('/')}/#/experiments/{info.experiment_id}/runs/{info.run_id}",
    }, indent=2))
    print(f"Logged run '{run_id}' to {tracking_uri}")


def upload_dir_to_s3(local_dir: Path, key_prefix: str) -> str:
    """Upload every file under local_dir to the configured S3/MinIO bucket under
    key_prefix/, preserving relative paths. Returns the s3:// URI of the prefix."""
    import boto3

    bucket = os.environ["S3_BUCKET"]
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        s3.create_bucket(Bucket=bucket)

    for path in sorted(local_dir.rglob("*")):
        if path.is_file():
            key = f"{key_prefix}/{path.relative_to(local_dir).as_posix()}"
            s3.upload_file(str(path), bucket, key)
    return f"s3://{bucket}/{key_prefix}/"


def tag_mlflow_remote(run_dir: Path, remote_uri: str) -> None:
    """Record the remote artifact URI on the run's MLflow run (if it was logged)."""
    ref = run_dir / "mlflow_run.json"
    if not ref.exists():
        return
    import mlflow

    info = json.loads(ref.read_text())
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.tracking.MlflowClient().set_tag(info["mlflow_run_id"], "remote_uri", remote_uri)


def docker_step(task_id: str, script: str) -> DockerOperator:
    """A DockerOperator that runs one pipeline step inside the project image.

    Docker-out-of-Docker: the container drives the *host* daemon via the mounted
    socket (the agent/eval spin up per-instance SWE-bench containers), and run
    artifacts land on the host through the bind-mounted runs/ directory. The only
    templated value is the run_id pulled from prepare_run's XCom.
    """
    return DockerOperator(
        task_id=task_id,
        image=PIPELINE_IMAGE,
        command=[
            "python", f"pipeline/{script}",
            "--run-dir", CONTAINER_RUNS + "/{{ ti.xcom_pull(task_ids='prepare_run') }}",
        ],
        docker_url="unix://var/run/docker.sock",
        auto_remove="success",   # older docker providers want auto_remove=True
        mount_tmp_dir=False,
        mounts=[
            # DooD: the container's docker CLI talks to the host daemon
            Mount(source="/var/run/docker.sock", target="/var/run/docker.sock", type="bind"),
            # persist run artifacts on the host (source must be a host path)
            Mount(source=str(HOST_RUNS), target=CONTAINER_RUNS, type="bind"),
        ],
        environment={
            "NEBIUS_API_KEY": os.environ.get("NEBIUS_API_KEY", ""),
            "MSWEA_COST_TRACKING": "ignore_errors",
        },
        execution_timeout=STEP_TIMEOUT,
    )


@dag(
    dag_id="evaluate_agent",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    # retries/retry_delay apply to every task; heavy steps also set execution_timeout below.
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    params={
        "split": Param("test", type="string"),
        "subset": Param("verified", type="string", enum=["verified", "lite", "full"]),
        "workers": Param(1, type="integer", minimum=1),
        "model": Param("nebius/moonshotai/Kimi-K2.6", type="string"),
        "task_slice": Param("0:1", type="string"),   # mini-swe-agent --slice
        "run_id": Param("", type="string"),           # empty -> auto-generate
        "cost_limit": Param(0, type=["integer", "number"]),
    },
)
def evaluate_agent():

    @task
    def prepare_run() -> str:
        ctx = get_current_context()
        config = build_config(ctx)

        run_dir = RUNS_DIR / config["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))

        print(f"Prepared run dir: {run_dir}")
        return config["run_id"]   # run_id flows downstream; containers build their own paths

    # Heavy, environment-sensitive steps run in isolated containers (DockerOperator).
    run_agent = docker_step("run_agent", "run_agent.py")
    run_eval = docker_step("run_eval", "run_eval.py")

    @task
    def summarize_and_log(rid: str) -> str:
        run_id = rid   # param can't be named 'run_id' — it's a reserved TaskFlow context key
        run_dir = RUNS_DIR / run_id
        eval_dir = run_dir / "run-eval"
        config = json.loads((run_dir / "config.json").read_text())

        # The summary report is <model>.<run_id>.json; rglob finds it wherever it landed.
        reports = sorted(eval_dir.rglob(f"*.{run_id}.json"))
        if not reports:
            found = [p.name for p in eval_dir.iterdir()] if eval_dir.exists() else "run-eval missing"
            raise FileNotFoundError(
                f"No SWE-bench summary report (*.{run_id}.json) under {eval_dir}. Found: {found}"
            )
        report = json.loads(reports[0].read_text())

        submitted = report.get("submitted_instances", 0)
        resolved = report.get("resolved_instances", 0)
        metrics = {
            "run_id": run_id,
            "model": config["model"],
            "dataset_name": config["dataset_name"],
            "total_instances": report.get("total_instances", 0),
            "submitted_instances": submitted,
            "completed_instances": report.get("completed_instances", 0),
            "resolved_instances": resolved,
            "unresolved_instances": report.get("unresolved_instances", 0),
            "empty_patch_instances": report.get("empty_patch_instances", 0),
            "error_instances": report.get("error_instances", 0),
            "resolve_rate": (resolved / submitted) if submitted else 0.0,
        }

        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"Metrics: {metrics}")

        # Log to MLflow in-process (mlflow-skinny client in the Airflow env).
        log_to_mlflow(run_dir, config, metrics)

        return run_id

    @task
    def finalize_run(rid: str) -> str:
        run_id = rid   # param can't be named 'run_id' — it's a reserved TaskFlow context key
        run_dir = RUNS_DIR / run_id
        manifest = build_manifest(run_dir)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"Wrote manifest: {run_dir / 'manifest.json'}")
        return run_id

    @task(execution_timeout=timedelta(minutes=15))
    def upload_artifacts(rid: str) -> str:
        run_id = rid
        run_dir = RUNS_DIR / run_id

        if not (os.environ.get("S3_ENDPOINT_URL") and os.environ.get("S3_BUCKET")):
            print("Object storage not configured (S3_ENDPOINT_URL / S3_BUCKET unset); skipping upload.")
            return run_id

        remote_uri = f"s3://{os.environ['S3_BUCKET']}/{run_id}/"
        # Record the remote URI in manifest.json BEFORE uploading, so the uploaded copy has it.
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifact_location"]["remote"] = remote_uri
        manifest_path.write_text(json.dumps(manifest, indent=2))

        upload_dir_to_s3(run_dir, run_id)
        tag_mlflow_remote(run_dir, remote_uri)

        print(f"Uploaded run '{run_id}' -> {remote_uri}")
        return run_id

    run_id = prepare_run()
    run_id >> run_agent >> run_eval          # prepare -> agent -> eval (file handoff on disk)
    summary = summarize_and_log(run_id)
    run_eval >> summary                      # summarize waits for eval to finish
    finalized = finalize_run(summary)
    upload_artifacts(finalized)


evaluate_agent()
