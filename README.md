# Project 2 — DNN Inference Benchmarking on GCP

Performance benchmarking of ResNet-50 inference across four GCP deployment configurations (GPU VM, CPU VM, Cloud Run, Cloud Functions), with a confidential-computing overhead study layered on top.

This repository holds the **Cloud Run slice** and the **shared benchmarking harness** that all four teammates run against their own deployments.

**Team:** Aashir Khan, Ethan Chang, Yang-Jung (Eric) Chen, Shopnil Shahriar
**Model:** ResNet-50, ONNX, FP32, batch size 1
**Serving framework:** Triton Inference Server (`nvcr.io/nvidia/tritonserver:24.01-py3`)
**Region:** us-central1
**Project:** `ancient-acumen-486002-j4`

---

## Repository layout

```
Project2/
├── model/         # ResNet-50 → ONNX export (canonical model artifact)
├── cloudrun/      # Cloud Run deployment (Triton + ResNet-50 in a container)
├── harness/       # Shared benchmarking harness (used by all 4 teammates)
└── results/       # Raw per-request CSVs, organized by config
```

The `resnet50.onnx` binary itself is **not committed** (gitignored). It is reproduced from [model/export_model.py](model/export_model.py) and integrity-checked against [model/resnet50.onnx.sha256](model/resnet50.onnx.sha256).

---

## [model/](model/) — Canonical model artifact

Produces a single self-contained ONNX file used by every deployment configuration in the study.

| File | Purpose |
|---|---|
| [model/export_model.py](model/export_model.py) | Loads `ResNet50_Weights.IMAGENET1K_V2` from torchvision, exports to ONNX (opset 18, single-file via `dynamo=False`), structurally validates with `onnx.checker`, numerically validates against PyTorch outputs, writes SHA-256. |
| [model/resnet50.onnx.sha256](model/resnet50.onnx.sha256) | Committed integrity hash. Lets any teammate verify they have the exact same `.onnx` everyone else built against. |

**Run it:**

```bash
python model/export_model.py
```

Outputs `resnet50.onnx` at the repo root (~98 MB). Verify integrity:

```bash
shasum -a 256 -c model/resnet50.onnx.sha256
```

**Tensor contract** (must match `config.pbtxt` and the harness verbatim):
- Input:  `input`, FP32, shape `[1, 3, 224, 224]`
- Output: `output`, FP32, shape `[1, 1000]`

We deliberately skipped a GCS bucket for the model. The export is deterministic from the script + hash, removing a redundant cloud-auth dependency for a one-time fixed artifact.

---

## [cloudrun/](cloudrun/) — Cloud Run deployment

Builds a Triton Inference Server container with ResNet-50 baked into the image, deploys to Cloud Run as the "container" configuration in the four-way comparison.

| File | Purpose |
|---|---|
| [cloudrun/Dockerfile](cloudrun/Dockerfile) | `FROM nvcr.io/nvidia/tritonserver:24.01-py3`, copies `model_repository/` to `/models`, exposes ports 8000 (HTTP) / 8001 (gRPC) / 8002 (metrics). |
| [cloudrun/model_repository/resnet50/config.pbtxt](cloudrun/model_repository/resnet50/config.pbtxt) | Triton model config — declares input/output names, FP32 dtype, fixed shape, `KIND_AUTO` instance group. |
| [cloudrun/model_repository/resnet50/1/model.onnx](cloudrun/model_repository/resnet50/1/) | The exported ResNet-50 ONNX, copied from the repo root after running `export_model.py`. Gitignored. |
| [cloudrun/test_inference.py](cloudrun/test_inference.py) | Smoke test — sends one synthetic inference request to a deployed endpoint and prints top-5 predictions + latency. |
| [cloudrun/.gcloudignore](cloudrun/.gcloudignore) | Controls what `gcloud builds` uploads. Explicitly allows `*.onnx` so the Dockerfile `COPY` step has the model in build context. |
| [cloudrun/CLAUDE.md](cloudrun/CLAUDE.md) | Detailed deployment notes: GCP environment, required Cloud Run flags, pitfalls (autoscaling variance, CPU throttling, concurrency mismatch), build/deploy steps. |

**Build and deploy (summary — see [cloudrun/CLAUDE.md](cloudrun/CLAUDE.md) for the full procedure):**

```bash
# 1. Export model and stage it for the image
python model/export_model.py
cp resnet50.onnx cloudrun/model_repository/resnet50/1/model.onnx

# 2. Build, push, deploy
cd cloudrun
docker build -t triton-resnet50:v1 .
# tag + push to Artifact Registry: us-central1-docker.pkg.dev/ancient-acumen-486002-j4/benchmark-images/triton-resnet50:v1
gcloud run deploy triton-resnet50 \
  --image us-central1-docker.pkg.dev/ancient-acumen-486002-j4/benchmark-images/triton-resnet50:v1 \
  --region us-central1 \
  --port 8000 \
  --cpu 4 --memory 8Gi \
  --min-instances 1 --max-instances 1 \
  --concurrency 100 \
  --timeout 60 \
  --no-cpu-throttling \
  --allow-unauthenticated
```

**Smoke test the live endpoint:**

```bash
python cloudrun/test_inference.py --endpoint https://triton-resnet50-xxxxx-uc.a.run.app
```

**Why these flags matter** (full discussion in [cloudrun/CLAUDE.md](cloudrun/CLAUDE.md)):
- `--min-instances=1 --max-instances=1` — disable autoscaling so cold-starting a second instance doesn't add 5–30s of latency variance.
- `--concurrency=100` — Cloud Run's per-instance concurrency must be ≥ harness max concurrency, or requests queue at the frontend before reaching Triton.
- `--no-cpu-throttling` — required for clean CPU utilization metrics; without it Cloud Run throttles CPU between requests.

---

## [harness/](harness/) — Shared benchmarking harness

The benchmarking program every teammate runs against their own deployment endpoint. Produces one CSV per `(config, concurrency, run)` combination — 20 CSVs per config (4 concurrency levels × 5 runs).

| File | Purpose |
|---|---|
| [harness/harness.py](harness/harness.py) | Python program. Spawns N concurrent threads via `ThreadPoolExecutor`, each sending sequential inference requests through `tritonclient[http]`. Discards warmup requests, records per-request latency, writes a CSV with one row per request. |
| [harness/run_all.sh](harness/run_all.sh) | Thin bash wrapper. Loops over `concurrency ∈ {1, 10, 50, 100}` × `run ∈ {1..5}` and invokes `harness.py` 20 times. Tracks failures and prints a re-run snippet for any that errored. |
| [harness/requirements.txt](harness/requirements.txt) | Pinned dependencies: `tritonclient[http]==2.43.0`, `numpy`, `Pillow`. |

**Setup:**

```bash
pip install -r harness/requirements.txt
```

**Single run:**

```bash
python harness/harness.py \
  --endpoint https://triton-resnet50-xxxxx-uc.a.run.app \
  --config-name cloudrun \
  --concurrency 10 \
  --run 1 \
  --requests 200 \
  --warmup 20 \
  --output results/cloudrun/
```

**Full 20-run protocol:**

```bash
bash harness/run_all.sh https://triton-resnet50-xxxxx-uc.a.run.app cloudrun
```

**CLI contract** (same across all 5 configs — `gpu`, `cpu`, `confidential`, `cloudrun`, `cloudfunction`):

| Flag | Default | Notes |
|---|---|---|
| `--endpoint` | (required) | `https://...` for Cloud Run, `http://IP:8000` for VMs. SSL inferred from scheme. |
| `--config-name` | (required) | One of the 5 configs above; used in the output filename. |
| `--concurrency` | (required) | One of `{1, 10, 50, 100}`. |
| `--run` | (required) | Run number `1..5`. |
| `--requests` | 200 | Per-client request count, including warmup. |
| `--warmup` | 20 | First N per client are discarded from the CSV. |
| `--output` | `results/` | Output directory for the CSV. |

**CSV schema** (one row per measured request):

```
request_id, client_id, send_ts, receive_ts, latency_ms, success, error
```

The harness intentionally does **not** compute p50/p95/p99 or throughput inline — only raw per-request data. Aggregation across all teammates' CSVs happens in a separate phase at the end of the project.

---

## [results/](results/) — Raw benchmark data

Per-config subdirectories with the 20 CSVs produced by `run_all.sh`. Naming:

```
results/<config>/results_<config>_<concurrency>_<run>.csv
```

Currently committed: [results/cloudrun/](results/cloudrun/) — 5 runs of concurrency 1 (runs 2–5 of higher concurrency to follow).

---

## End-to-end reproduction

```bash
# 1. Set up
python -m venv venv && source venv/bin/activate
pip install -r harness/requirements.txt

# 2. Build the model artifact
python model/export_model.py
shasum -a 256 -c model/resnet50.onnx.sha256

# 3. Deploy (Cloud Run example — see cloudrun/CLAUDE.md for full procedure)
cp resnet50.onnx cloudrun/model_repository/resnet50/1/model.onnx
# ... build, push, deploy ...

# 4. Smoke test
python cloudrun/test_inference.py --endpoint <your-endpoint>

# 5. Run the full benchmark protocol
bash harness/run_all.sh <your-endpoint> cloudrun
```
