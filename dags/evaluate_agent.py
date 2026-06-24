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
        # TODO 1.3: run mini-extra swebench into <run_dir>/run-agent, return preds.json path
        ...

    @task
    def run_eval(run_dir: str, preds_path: str) -> str:
        # TODO 1.4: run swebench eval into <run_dir>/run-eval, return the eval dir
        ...

    @task
    def summarize_and_log(run_dir: str, eval_dir: str) -> None:
        # TODO 1.5: parse the summary report -> metrics.json
        ...

    rd = prepare_run()
    preds = run_agent(rd)
    ev = run_eval(rd, preds)
    summarize_and_log(rd, ev)


evaluate_agent()
