"""Log a completed run's params, metrics, and key artifacts to MLflow.

Reads ``runs/<run-id>/config.json`` and ``metrics.json`` plus the SWE-bench
summary report, then logs them to the tracking server at ``$MLFLOW_TRACKING_URI``.

Runs in the project venv (where ``mlflow`` is installed) and is invoked by the
Airflow ``summarize_and_log`` task via ``uv run``. It is also runnable on its own,
so a finished run can be (re-)logged independently of Airflow:

    uv run python pipeline/log_mlflow.py --run-dir runs/<run-id>
"""
import argparse
import json
import os
from pathlib import Path

import mlflow

PARAM_KEYS = [
    "run_id",
    "airflow_run_id",
    "split",
    "subset",
    "dataset_name",
    "model",
    "task_slice",
    "workers",
    "cost_limit",
    "git_sha",
]
METRIC_KEYS = [
    "total_instances",
    "submitted_instances",
    "completed_instances",
    "resolved_instances",
    "unresolved_instances",
    "empty_patch_instances",
    "error_instances",
    "resolve_rate",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Path to runs/<run-id>/")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    run_id = config["run_id"]

    # The SWE-bench summary report lives at the top of run-eval/.
    eval_dir = run_dir / "run-eval"
    reports = sorted(eval_dir.glob(f"*.{run_id}.json"))
    report_path = reports[0] if reports else None

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "swebench-eval"))

    with mlflow.start_run(run_name=run_id):
        mlflow.log_params({k: config.get(k) for k in PARAM_KEYS})
        mlflow.log_metrics({k: metrics[k] for k in METRIC_KEYS if k in metrics})

        # Record where the full artifacts live (local now; S3 URI added in Phase 6).
        mlflow.set_tag("artifact_dir", str(run_dir.resolve()))

        # Attach the small, high-value files directly to the MLflow run.
        for f in (run_dir / "config.json", run_dir / "metrics.json", report_path):
            if f and Path(f).exists():
                mlflow.log_artifact(str(f))

    print(f"Logged run '{run_id}' to {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
