# Evaluation pipeline for coding-agent experiments — Report

An Airflow pipeline that runs [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
on a slice of [SWE-bench](https://www.swebench.com/) tasks, evaluates the produced
patches with the SWE-bench harness, and captures every run as a reproducible,
structured artifact set with MLflow tracking and Object Storage upload.

---

## Architecture

The DAG `evaluate_agent` (`dags/evaluate_agent.py`) is a six-task pipeline:

| Task | Runs in | Produces |
|---|---|---|
| `prepare_run` | Airflow (host) | `runs/<id>/config.json` — frozen params + provenance (git SHA, timestamps, dataset) |
| `run_agent` | **DockerOperator** (`mlops-pipeline` image) | `run-agent/preds.json` + per-instance trajectories |
| `run_eval` | **DockerOperator** | `run-eval/logs/…` + `run-eval/reports/<model>.<id>.json` |
| `summarize_and_log` | Airflow | `metrics.json` + an MLflow run (params, metrics, artifacts) |
| `finalize_run` | Airflow | `manifest.json` — index of the run |
| `upload_artifacts` | Airflow | uploads `runs/<id>/` to Object Storage, records the `s3://` URI |

**Design principle:** the heavy, environment-sensitive steps (`run_agent`, `run_eval`)
run in **isolated containers** via `DockerOperator`. Because mini-swe-agent and the
SWE-bench harness both spin up per-instance Docker containers, those steps use
**Docker-out-of-Docker** — the project container drives the host daemon through a
mounted socket, and run artifacts persist on the host through a bind-mounted `runs/`
directory. Lightweight orchestration (config, metrics, manifest, upload) stays in the
Airflow process.

```
prepare_run ─▶ run_agent ─▶ run_eval ─▶ summarize_and_log ─▶ finalize_run ─▶ upload_artifacts
  (host)        (docker)     (docker)        (host)              (host)           (host)
                   └── DooD: /var/run/docker.sock + host runs/ bind-mount ──┘
```

In docker-compose mode the whole control plane is containerized:

```
            compose network
  ┌──────────┬──────────┬─────────┬─────────┐
  │ postgres │ airflow  │ mlflow  │  minio  │
  └──────────┴────┬─────┴────┬────┴────┬────┘
                  │ socket   │ :5000   │ :9000 (S3) / :9001 (console)
                  ▼
       host daemon spawns SIBLING containers (mlops-pipeline:latest)
       run_agent / run_eval  — bind-mount the host runs/ directory
```

---

## Deployment

### Prerequisites
- A Linux host with Docker (8 CPU / 32 GB recommended), `uv`, and a `NEBIUS_API_KEY`.
- Build the project image used by the agent/eval steps and the MLflow server:
  ```bash
  docker build -t mlops-pipeline:latest .
  ```

### Option A — Standalone (development)
```bash
uv sync
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000   # terminal 1
set -a; source .env; set +a
bash run-airflow-standalone.sh                                                            # terminal 2
```
Airflow at `:8080` (admin/admin), MLflow at `:5000`. Object-storage upload is skipped
unless you point `S3_*` env vars at a MinIO/S3 you run yourself.

### Option B — docker-compose (production-style)
Bundles Postgres + Airflow + MLflow + MinIO. Required `.env` keys:
```
NEBIUS_API_KEY=...
HOST_PROJECT_DIR=<absolute path to this repo>     # id-independent: $(pwd)
AIRFLOW_UID=<id -u>
DOCKER_GID=<getent group docker | cut -d: -f3>
```
Then:
```bash
mkdir -p logs
docker compose up -d --build
```
- Airflow `:8080`, MLflow `:5000`, MinIO console `:9001` (minioadmin / minioadmin).
- The DAG logs to `http://mlflow:5000` and uploads to the `mlops-runs` bucket — all wired
  by the compose file (no extra `.env` for MLflow/S3).

---

## Configuration (DAG parameters)

No experiment values are hard-coded; everything is an Airflow param surfaced in the
"Trigger DAG w/ config" form:

| Param | Default | Meaning |
|---|---|---|
| `split` | `test` | SWE-bench split |
| `subset` | `verified` | `verified` / `lite` / `full` → mapped to the dataset name |
| `workers` | `1` | parallelism for agent and eval |
| `model` | `nebius/moonshotai/Kimi-K2.6` | model id passed to mini-swe-agent |
| `task_slice` | `0:1` | slice of instances to run (e.g. `0:3`) |
| `run_id` | `` (auto) | run identifier / folder name; empty → UTC timestamp |
| `cost_limit` | `0` | per-instance cost cap (`0` keeps the config default) |

## Triggering a run
Airflow UI → `evaluate_agent` → **Trigger w/ config** → set params → **Trigger**.
Watch the six tasks; `run_agent`/`run_eval` stream their container logs.

---

## Run artifact layout

Every run is a self-contained, portable folder:

```
runs/<run-id>/
  config.json          # frozen params + provenance (git SHA, dataset, timestamps)
  run-agent/
    preds.json         # {instance_id: {model_name_or_path, instance_id, model_patch}}
    <instance>/<instance>.traj.json    # agent trajectory
  run-eval/
    logs/run_evaluation/<id>/<model>/<instance>/   # eval.sh, patch.diff, report.json, test_output.txt
    reports/<model>.<id>.json                       # SWE-bench summary
  metrics.json         # resolve_rate + instance counts
  mlflow_run.json      # link back to the MLflow run (id, experiment, URL)
  manifest.json        # index of everything above + remote artifact URI
```

`manifest.json` uses **relative** pointers, so the folder can be moved, zipped, or
uploaded and the links still resolve — that's what makes a run reconstructable from one
directory or one URI.

## Reproducing / inspecting a run by `run-id`
- **Understand a past run:** read `runs/<id>/manifest.json` (or pull `s3://mlops-runs/<id>/`)
  — it points to the config, predictions, trajectories, eval logs, metrics, and MLflow run.
- **Re-run the same configuration:** trigger the DAG with the same params (the `config.json`
  records exactly what produced a result, including the pipeline git SHA).

---

## MLflow tracking
All runs log to the `swebench-eval` experiment: params (`model`, `subset`, `dataset_name`,
`task_slice`, `workers`, `cost_limit`, `git_sha`, …), metrics (`resolve_rate` and the
instance counts), the key files as artifacts, an `artifact_dir` tag, and — after upload —
a `remote_uri` tag. Multiple runs are directly comparable in the UI (Runs tab → select →
**Compare**).

## Object storage
The `upload_artifacts` task pushes the full run folder to the `mlops-runs` bucket
(`s3://mlops-runs/<id>/…`), preserving paths, and records the URI in `manifest.json`
(`artifact_location.remote`) and on the MLflow run. In compose this is the bundled MinIO;
the same code works against real S3 by changing the `S3_*` env vars.

---

## Example completed run
A `task_slice=0:1` run on `SWE-bench_Verified` solved `astropy__astropy-12907`
(`resolved_instances: 1`, `submitted_instances: 1`, **`resolve_rate: 1.0`**). The run was
logged to MLflow and uploaded to `s3://mlops-runs/<id>/` — see the screenshots below.

## Screenshots
- `screenshots/airflow_dag.png` — the `evaluate_agent` DAG completing (all six tasks green).
- `screenshots/mlflow_runs.png` — logged runs with metrics, comparable in the MLflow UI.
- `screenshots/object_storage_artifacts.png` — the `mlops-runs` bucket with an uploaded run.
