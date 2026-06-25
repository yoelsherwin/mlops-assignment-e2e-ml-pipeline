import os
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from airflow.decorators import dag, task
from airflow.models.param import Param  # available on Airflow 2.x and 3.x

# get_current_context moved to the Task SDK in Airflow 3.x; fall back for 2.x.
try:
    from airflow.sdk import get_current_context  # Airflow 3.x
except ImportError:  # Airflow 2.x
    from airflow.operators.python import get_current_context

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
CONFIG_YAML = PROJECT_ROOT / "config" / "swebench.yaml"  # vendored mini-swe-agent config (pinned in-repo)
LOG_MLFLOW_SCRIPT = PROJECT_ROOT / "pipeline" / "log_mlflow.py"  # runs in the project venv via `uv run`

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
        return str(run_dir)

    @task
    def run_agent(run_dir: str) -> str:
        run_dir = Path(run_dir)
        config = json.loads((run_dir / "config.json").read_text())   # frozen config = source of truth

        out_dir = run_dir / "run-agent"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "uv", "run", "mini-extra", "swebench",
            "--subset", config["subset"],
            "--split", config["split"],
            "--model", config["model"],
            "--slice", config["task_slice"],
            "--config", str(CONFIG_YAML),       # vendored copy; -c replaces the packaged default
            "--workers", str(config["workers"]),
            "-o", str(out_dir),
        ]
        # cost_limit has no batch CLI flag; it's a merged config override.
        # 0 -> keep the config's built-in limit; >0 -> override it for this run.
        if config["cost_limit"] and float(config["cost_limit"]) > 0:
            cmd += ["--config", f"agent.cost_limit={config['cost_limit']}"]

        env = {**os.environ, "MSWEA_COST_TRACKING": "ignore_errors"}
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)   # red on failure

        preds_path = out_dir / "preds.json"
        if not preds_path.exists():
            raise FileNotFoundError(f"agent did not produce {preds_path}")
        return str(preds_path)

    @task
    def run_eval(run_dir: str, preds_path: str) -> str:
        run_dir = Path(run_dir)
        config = json.loads((run_dir / "config.json").read_text())

        eval_dir = run_dir / "run-eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "uv", "run", "python", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", config["dataset_name"],
            "--predictions_path", str(preds_path),
            "--max_workers", str(config["workers"]),
            "--run_id", config["run_id"],
        ]
        # SWE-bench writes logs/ and the <model>.<run_id>.json summary into its CWD,
        # so run it *inside* run-eval/ to keep every eval artifact in one place.
        subprocess.run(cmd, cwd=eval_dir, check=True)

        return str(eval_dir)

    @task
    def summarize_and_log(run_dir: str, eval_dir: str) -> None:
        run_dir = Path(run_dir)
        eval_dir = Path(eval_dir)
        config = json.loads((run_dir / "config.json").read_text())
        run_id = config["run_id"]

        # SWE-bench writes the summary as <model>.<run_id>.json at the top of eval_dir.
        reports = sorted(eval_dir.glob(f"*.{run_id}.json"))
        if not reports:
            raise FileNotFoundError(
                f"No SWE-bench summary report (*.{run_id}.json) in {eval_dir}. "
                f"Found: {[p.name for p in eval_dir.iterdir()]}"
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

        # Log to MLflow via the project venv (mlflow isn't in the Airflow tool env).
        subprocess.run(
            ["uv", "run", "python", str(LOG_MLFLOW_SCRIPT), "--run-dir", str(run_dir)],
            cwd=PROJECT_ROOT,
            check=True,
        )

    rd = prepare_run()
    preds = run_agent(rd)
    ev = run_eval(rd, preds)
    summarize_and_log(rd, ev)


evaluate_agent()
