"""Run mini-swe-agent on a SWE-bench slice for one prepared run.

Reads ``runs/<id>/config.json`` and writes predictions + trajectories into
``runs/<id>/run-agent/``. Designed to run *inside* the project Docker image
(the venv is on PATH there), invoked by the Airflow ``run_agent`` DockerOperator:

    python pipeline/run_agent.py --run-dir /mlops-assignment/runs/<id>

It also runs on the host via ``uv run python pipeline/run_agent.py ...``.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_YAML = PROJECT_ROOT / "config" / "swebench.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Path to runs/<run-id>/")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())   # frozen config = source of truth

    out_dir = run_dir / "run-agent"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mini-extra", "swebench",
        "--subset", config["subset"],
        "--split", config["split"],
        "--model", config["model"],
        "--slice", config["task_slice"],
        "--config", str(CONFIG_YAML),       # vendored copy; -c replaces the packaged default
        "--workers", str(config["workers"]),
        "-o", str(out_dir),
    ]
    # cost_limit has no batch CLI flag; 0 keeps the config default, >0 overrides it.
    if config["cost_limit"] and float(config["cost_limit"]) > 0:
        cmd += ["--config", f"agent.cost_limit={config['cost_limit']}"]

    env = {**os.environ, "MSWEA_COST_TRACKING": "ignore_errors"}
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)

    preds_path = out_dir / "preds.json"
    if not preds_path.exists():
        raise FileNotFoundError(f"agent did not produce {preds_path}")
    print(f"preds: {preds_path}")


if __name__ == "__main__":
    main()
