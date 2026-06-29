import os
import json
import subprocess
from datetime import datetime, timezone
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
LOG_MLFLOW_SCRIPT = PROJECT_ROOT / "pipeline" / "log_mlflow.py"  # runs in the project venv via `uv run`

# Image built from the project Dockerfile; the agent/eval steps run inside it.
PIPELINE_IMAGE = os.environ.get("PIPELINE_IMAGE", "mlops-pipeline:latest")
# Paths *inside* the container (Dockerfile WORKDIR is /mlops-assignment).
CONTAINER_ROOT = "/mlops-assignment"
CONTAINER_RUNS = f"{CONTAINER_ROOT}/runs"

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
            Mount(source=str(RUNS_DIR), target=CONTAINER_RUNS, type="bind"),
        ],
        environment={
            "NEBIUS_API_KEY": os.environ.get("NEBIUS_API_KEY", ""),
            "MSWEA_COST_TRACKING": "ignore_errors",
        },
    )


@dag(
    dag_id="evaluate_agent",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
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

        # Log to MLflow on the host via the project venv (mlflow isn't in the Airflow env,
        # and host->localhost:5000 avoids container networking).
        subprocess.run(
            ["uv", "run", "python", str(LOG_MLFLOW_SCRIPT), "--run-dir", str(run_dir)],
            cwd=PROJECT_ROOT,
            check=True,
        )

        return run_id

    @task
    def finalize_run(rid: str) -> str:
        run_id = rid   # param can't be named 'run_id' — it's a reserved TaskFlow context key
        run_dir = RUNS_DIR / run_id
        manifest = build_manifest(run_dir)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"Wrote manifest: {run_dir / 'manifest.json'}")
        return run_id

    run_id = prepare_run()
    run_id >> run_agent >> run_eval          # prepare -> agent -> eval (file handoff on disk)
    summary = summarize_and_log(run_id)
    run_eval >> summary                      # summarize waits for eval to finish
    finalize_run(summary)


evaluate_agent()
