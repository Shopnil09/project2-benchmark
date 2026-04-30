# Project 2 — DNN Inference Benchmarking on GCP

Performance benchmarking of ResNet-50 inference across five GCP deployment configurations: GPU VM, CPU VM, Confidential VM, Cloud Run, and Cloud Functions. Includes a confidential-computing overhead study (AMD SEV) and an interactive results dashboard.

**Team:** Aashir Khan, Ethan Chang, Yang-Jung (Eric) Chen, Shopnil Shahriar
**Model:** ResNet-50, ONNX, FP32, batch size 1
**Serving framework:** Triton Inference Server (`nvcr.io/nvidia/tritonserver:24.01-py3`)
**Region:** us-central1
**Project:** `ancient-acumen-486002-j4`

**Dashboard:** https://shopnil09.github.io/project2-benchmark/

---

## Repository layout

```
project2-benchmark/
├── index.html              # Interactive results dashboard (Chart.js)
├── model/                  # ResNet-50 → ONNX export (canonical artifact)
├── cloudrun/               # Cloud Run deployment (Triton in a container)
├── cloudfunction/          # Cloud Functions 2nd gen deployment (onnxruntime adapter)
├── confidential/           # CPU VM + Confidential VM deployments (AMD SEV study)
├── harness/                # Shared benchmarking harness (used by all configs)
├── results/
│   ├── cloudrun/           # 20 CSVs + NOTES.md (Shopnil Shahriar)
│   ├── cloudfunction/      # 20 CSVs + NOTES.md (Ethan Chang)
│   └── gpu/                # 20 CSVs + notes_gpu.json (Aashir Khan)
├── confidential/results/
│   ├── cpu/                # 20 CSVs (Yang-Jung Chen)
│   └── confidential/       # 20 CSVs (Yang-Jung Chen)
```

The `resnet50.onnx` binary is **not committed** (gitignored, ~98 MB). Reproduce it from [model/export_model.py](model/export_model.py) and verify against [model/resnet50.onnx.sha256](model/resnet50.onnx.sha256).

---

## Benchmark status

All five configurations are fully benchmarked. Each ran the complete protocol: concurrency ∈ {1, 10, 50, 100} × 5 runs × 200 requests per client (20 warmup discarded) = 20 CSVs per config.

| Config | Owner | Hardware | Error-free runs | Notes |
|---|---|---|---|---|
| GPU VM | Aashir Khan | NVIDIA L4, g2-standard-4 | 20/20 | L4 substituted for T4 (T4 exhausted in all us-central1 zones) |
| CPU VM | Yang-Jung Chen | n2d-standard-4 | 20/20 | Baseline for TEE overhead study |
| Confidential VM | Yang-Jung Chen | n2d-standard-4 + AMD SEV | 20/20 | +0.7% latency vs CPU VM, +6% cost/hr |
| Cloud Run | Shopnil Shahriar | 4 vCPU / 8 GB (managed) | 20/20 | |
| Cloud Functions | Ethan Chang | 4 vCPU / 8 GB (managed) | 15/20 | C=100 runs exceed 5% error threshold (structural, not transient) |

---

## Key results

| Config | p50 @ C=1 | p50 @ C=10 | Throughput @ C=100 | Cost / 1k req |
|---|---|---|---|---|
| GPU (L4) | 4.7 ms | 38.4 ms | 205 req/s | $0.0026 |
| CPU VM | 55.4 ms | 513 ms | 17.8 req/s | $0.0026 |
| Confidential VM | 55.6 ms | 516 ms | 19.7 req/s | $0.0025 |
| Cloud Run | 74.3 ms | 532 ms | 16.2 req/s | $0.0616 |
| Cloud Function | 205 ms | 480 ms | 15.2 req/s | $0.0560 |

---

## [model/](model/) — Canonical model artifact

| File | Purpose |
|---|---|
| [model/export_model.py](model/export_model.py) | Exports `ResNet50_Weights.IMAGENET1K_V2` to ONNX (opset 18), validates structurally and numerically, writes SHA-256. |
| [model/resnet50.onnx.sha256](model/resnet50.onnx.sha256) | Committed integrity hash — all teammates verified the same artifact. |

```bash
python model/export_model.py
shasum -a 256 -c model/resnet50.onnx.sha256
```

**Tensor contract:** input `[1, 3, 224, 224]` FP32 → output `[1, 1000]` FP32.

---

## [cloudrun/](cloudrun/) — Cloud Run (Shopnil Shahriar)

Triton Inference Server in a Docker container deployed to Cloud Run.

| File | Purpose |
|---|---|
| [cloudrun/Dockerfile](cloudrun/Dockerfile) | `FROM nvcr.io/nvidia/tritonserver:24.01-py3`, copies model repository, exposes ports 8000/8001/8002. |
| [cloudrun/model_repository/resnet50/config.pbtxt](cloudrun/model_repository/resnet50/config.pbtxt) | Triton model config — FP32, `KIND_AUTO` instance group. |
| [cloudrun/test_inference.py](cloudrun/test_inference.py) | Smoke test — one synthetic request, prints top-5 + latency. |
| [cloudrun/CLAUDE.md](cloudrun/CLAUDE.md) | Full deployment notes and pitfalls. |

```bash
python model/export_model.py
cp resnet50.onnx cloudrun/model_repository/resnet50/1/model.onnx
bash harness/run_all.sh <cloud-run-url> cloudrun
```

---

## [cloudfunction/](cloudfunction/) — Cloud Functions 2nd gen (Ethan Chang)

onnxruntime-based inference adapter that speaks the Triton KServe v2 HTTP protocol, allowing the shared harness to target it without modification.

| File | Purpose |
|---|---|
| [cloudfunction/main.py](cloudfunction/main.py) | HTTP adapter: loads ONNX model at cold start, routes Triton v2 requests to `onnxruntime`. |
| [cloudfunction/deploy.sh](cloudfunction/deploy.sh) | Deploys as Cloud Functions 2nd gen (4 vCPU, 8 GiB, min-instances=1). |
| [cloudfunction/test_inference.py](cloudfunction/test_inference.py) | Smoke test. |
| [cloudfunction/CLAUDE.md](cloudfunction/CLAUDE.md) | Deployment notes, cold-start methodology, pitfalls. |

```bash
python model/export_model.py
cp resnet50.onnx cloudfunction/
bash cloudfunction/deploy.sh
bash harness/run_all.sh <function-url> cloudfunction
```

---

## [confidential/](confidential/) — CPU VM + Confidential VM (Yang-Jung Chen)

Paired deployment of standard and AMD SEV-encrypted VMs on identical hardware (`n2d-standard-4`) to isolate TEE overhead.

| File | Purpose |
|---|---|
| [confidential/deploy_vm.sh](confidential/deploy_vm.sh) | Provisions both VMs. |
| [confidential/setup_vm.sh](confidential/setup_vm.sh) | Installs Docker, pulls Triton image, starts the server. |
| [confidential/README.md](confidential/README.md) | Full setup and benchmarking procedure. |

Results show AMD SEV adds ~3 ms p50 overhead at C=10 (0.7%) and ~6% hourly cost — effectively free in performance terms for this workload.

---

## GPU VM (Aashir Khan)

NVIDIA L4 GPU on a `g2-standard-4` Compute Engine instance running Triton Inference Server with the ONNX backend.

**Deviations from protocol:**
- NVIDIA L4 used instead of T4 — T4 was exhausted across all us-central1 zones and tested US regions at provisioning time
- `g2-standard-4` instead of `n1-standard-4` — required machine type for L4
- Harness ran on the VM (localhost:8000) — org policy blocks external IPs; this eliminates network latency from measurements

---

## [harness/](harness/) — Shared benchmarking harness

| File | Purpose |
|---|---|
| [harness/harness.py](harness/harness.py) | Spawns N concurrent threads via `ThreadPoolExecutor`, sends sequential inference requests via `tritonclient[http]`, records per-request latency to CSV. |
| [harness/run_all.sh](harness/run_all.sh) | Loops over concurrency ∈ {1, 10, 50, 100} × run ∈ {1..5}, invokes harness.py 20 times. |
| [harness/requirements.txt](harness/requirements.txt) | `tritonclient[http]==2.43.0`, `numpy`, `Pillow`. |

```bash
pip install -r harness/requirements.txt
bash harness/run_all.sh <endpoint-url> <config-name>
```

**CSV schema:** `request_id, client_id, send_ts, receive_ts, latency_ms, success, error`

---

## [results/](results/) — Raw benchmark data

```
results/<config>/results_<config>_<concurrency>_<run>.csv
confidential/results/<config>/results_<config>_<concurrency>_<run>.csv
```

20 CSVs per config (4 concurrency levels × 5 runs), 180 measured rows per CSV (200 requests − 20 warmup). Aggregated p50/p95/p99 and throughput are in the dashboard; raw per-request data is the canonical record.

---

## End-to-end reproduction

```bash
# 1. Set up
python -m venv venv && source venv/bin/activate
pip install -r harness/requirements.txt

# 2. Build the model artifact
python model/export_model.py
shasum -a 256 -c model/resnet50.onnx.sha256

# 3. Deploy your config (see per-config README/CLAUDE.md)

# 4. Run the full benchmark protocol
bash harness/run_all.sh <your-endpoint> <config-name>
```
