"""Evaluate predictions with the SWE-bench harness for one prepared run.

Reads ``runs/<id>/config.json`` + ``run-agent/preds.json`` and writes eval logs
and the summary report into ``runs/<id>/run-eval/``. Designed to run *inside*
the project Docker image, invoked by the Airflow ``run_eval`` DockerOperator:

    python pipeline/run_eval.py --run-dir /mlops-assignment/runs/<id>

It also runs on the host via ``uv run python pipeline/run_eval.py ...``.
"""
import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Path to runs/<run-id>/")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    run_id = config["run_id"]

    eval_dir = run_dir / "run-eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "swebench.harness.run_evaluation",
        "--dataset_name", config["dataset_name"],
        "--predictions_path", str(run_dir / "run-agent" / "preds.json"),
        "--max_workers", str(config["workers"]),
        "--run_id", run_id,
    ]
    # SWE-bench writes logs/ and the <model>.<run_id>.json summary into its CWD,
    # so run it *inside* run-eval/ to keep every eval artifact in one place.
    subprocess.run(cmd, cwd=eval_dir, check=True)

    # Tidy the summary report into run-eval/reports/ (logs/ already lands correctly).
    reports_dir = eval_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    for report in eval_dir.glob(f"*.{run_id}.json"):
        report.rename(reports_dir / report.name)

    print(f"eval done: {eval_dir}")


if __name__ == "__main__":
    main()
