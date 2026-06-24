import os
import json
import subprocess
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task
# Import path depends on your Airflow version — see the note below:
from airflow.models.param import Param
from airflow.operators.python import get_current_context

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"

# subset -> SWE-bench dataset name that the eval harness expects
DATASET_BY_SUBSET = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
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
        params = ctx["params"]
        # TODO 1.2: build config dict, create runs/<run-id>/, write config.json,
        #           return the run_dir path as a string
        ...

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