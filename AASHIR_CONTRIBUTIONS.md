# Aashir Khan — GPU Deployment Contributions

**Project:** DNN Inference Benchmarking on GCP  
**Role:** GPU instance deployment and benchmarking  
**Date:** 2026-04-28  
**Email:** ak5445@columbia.edu

---

## Summary

Provisioned a GCP Compute Engine GPU VM, deployed Triton Inference Server with the shared ResNet-50 ONNX model, and ran the full 20-run benchmark protocol across all four concurrency levels. All 20 runs completed with 0% error rate.

---

## GCP Environment Setup

**Installed gcloud CLI** via Homebrew on a Mac that had no prior GCP tooling.

**Authenticated** with ak5445@columbia.edu and configured the project `ancient-acumen-486002-j4`. Initially had only `roles/editor` access, which was insufficient to grant IAM roles. Was upgraded to `roles/owner` by Shopnil Shahriar. Self-granted `roles/iap.tunnelResourceAccessor` to enable IAP SSH tunneling (required because the GCP org policy `compute.vmExternalIpAccess` blocks external IPs on all VMs in this project).

---

## VM Provisioning

**GPU availability issue:** The protocol called for an `n1-standard-4 + NVIDIA T4` in `us-central1`. T4 GPUs were exhausted across all four `us-central1` zones (`a`, `b`, `c`, `f`) and all tested US regions (`us-east1`, `us-west1`) at time of provisioning. Substituted an **NVIDIA L4** on a `g2-standard-4` machine type, which was available in `us-central1-a`. This deviation is documented in `results/gpu/notes_gpu.json`.

**VM created:**
- Machine type: `g2-standard-4`
- GPU: 1x NVIDIA L4
- OS: Container-Optimized OS (`cos-stable`)
- Zone: `us-central1-a`
- Boot disk: 100 GB
- No external IP (org policy constraint) — SSH via IAP tunnel throughout

**Networking:** The project already had a Cloud Router (`benchmark-router`) and Cloud NAT configured by Shopnil, which gave the no-external-IP VM outbound internet access for pulling Docker images.

---

## GPU Driver Installation

Installed the NVIDIA driver on Container-Optimized OS using `cos-extensions install gpu`, which pulled and ran the `cos-gpu-installer:v2.6.1` container. Installed driver version **535.288.01** for the L4.

Configured the NVIDIA Container Runtime for Docker:

1. Registered `nvidia-container-runtime` in `/etc/docker/daemon.json` with `"default-runtime": "nvidia"` and restarted the Docker daemon.
2. The runtime defaulted to CDI mode, but no CDI spec existed yet. Used `nvidia-ctk cdi generate` with the library path `/var/lib/nvidia/lib64` to generate `/etc/cdi/nvidia.yaml`. This enumerated all GPU devices (`/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, etc.) and the full NVIDIA library set.

---

## Model Export

The ResNet-50 ONNX model was not committed to the repo (gitignored, ~98 MB). Ran `model/export_model.py` locally to produce `resnet50.onnx`.

**Python version issue:** The system Python (3.8.10 via pyenv) produced a segfault when installing the `onnx` package. Used Homebrew's Python 3.11 (`python3.11`) instead, installed `torch`, `torchvision`, `onnx`, and `onnxruntime`, then re-ran the export script successfully.

Verified the exported model against the committed SHA-256 hash:
```
38da5bc82ddcd2e3a2f9b511b02622ae9be5dc8a50263a4a8adbea14bed12f78  resnet50.onnx
```
Hash matched — same artifact as every other teammate.

---

## Triton Deployment

Copied to the VM via IAP SCP tunnel:
- `resnet50.onnx` → `~/model_repository/resnet50/1/model.onnx` (97.4 MB)
- `cloudrun/model_repository/resnet50/config.pbtxt` → `~/model_repository/resnet50/config.pbtxt`

The same `config.pbtxt` used by Shopnil's Cloud Run deployment was reused — `KIND_AUTO` resolved to GPU device 0 on the L4 as expected.

Pulled the Triton image: `nvcr.io/nvidia/tritonserver:24.01-py3` (digest `sha256:3380720761045fc16ba3bcb96cfa54034531fc302df54ecac6b2a4deeab07bbd`).

Started Triton:
```bash
docker run -d --name triton \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -v ~/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models \
               --strict-model-config=true \
               --allow-metrics=true --allow-http=true
```

Triton startup log confirmed model loaded on **GPU device 0**:
```
TRITONBACKEND_ModelInstanceInitialize: resnet50_0_0 (GPU device 0)
successfully loaded 'resnet50'
Started HTTPService at 0.0.0.0:8000
```

---

## Benchmarking

**Harness execution:** COS does not include Python or pip. The harness was run inside a `python:3.11-slim` Docker container on the VM with `--network=host`, connecting to Triton at `http://localhost:8000`. This means all measured latency is pure Triton inference latency with zero network component — a cleaner measurement than remote client benchmarking would have produced given the no-external-IP constraint.

Ran the full 20-run protocol:
- Concurrency levels: 1, 10, 50, 100
- 5 runs per level
- 200 requests per client per run (20 warmup discarded, 180 measured)
- 5-second pause between runs

**Results — all 20 runs, 0% error rate:**

| Concurrency | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) |
|---|---|---|---|---|
| 1 | 4.7 | 4.9 | 5.3 | ~200 |
| 10 | 38.1 | 38.4 | 38.5 | ~250 |
| 50 | 188.5 | 191.3 | 193.1 | ~256 |
| 100 | 376.0 | 380.3 | 382.5 | ~258 |

Latency scales linearly with concurrency because the model config has a single instance group (`count: 1`), so requests serialize through one GPU model instance. The p50 at concurrency=1 (4.7ms) vs Cloud Run CPU (≈130ms) demonstrates the GPU advantage: **~28x lower latency** at single-request load.

---

## Files Committed

```
results/gpu/
├── results_gpu_1_1.csv   through  results_gpu_1_5.csv    (5 files, 180 rows each)
├── results_gpu_10_1.csv  through  results_gpu_10_5.csv   (5 files, 1800 rows each)
├── results_gpu_50_1.csv  through  results_gpu_50_5.csv   (5 files, 9000 rows each)
├── results_gpu_100_1.csv through  results_gpu_100_5.csv  (5 files, 18000 rows each)
└── notes_gpu.json        (instance config, deviations, GCP pricing)

AASHIR_CONTRIBUTIONS.md  (this file)
```

---

## Protocol Deviations

| Deviation | Reason |
|---|---|
| NVIDIA L4 instead of T4 | T4 exhausted across all us-central1 zones and tested US regions at time of provisioning |
| `g2-standard-4` instead of `n1-standard-4` | Required machine type for L4 GPU |
| Harness ran on VM (localhost) not remote client | Org policy blocks external IPs; IAP tunnel is too slow for high-concurrency load generation |
| Model from export script, not GCS bucket | Team decision documented in `cloudrun/CLAUDE.md` |
