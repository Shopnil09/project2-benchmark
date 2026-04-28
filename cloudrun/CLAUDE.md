# CLAUDE.md — Container Deployment (Cloud Run)

**Owner:** Shopnil Shahriar
**Project:** DNN Inference Benchmarking on GCP — Container slice
**Model:** ResNet-50 (ONNX, FP32, batch size 1)
**Serving framework:** Triton Inference Server
**Region:** us-central1

---

## GCP environment verified (April 2026)

- **Project:** `ancient-acumen-486002-j4`
- **VPC:** Default VPC exists in `AUTO` subnet mode → auto-created subnets in every region including `us-central1`. No custom networking needed.
- **Org policies:** No restrictive org policies enforced at the project level (verified via `gcloud resource-manager org-policies list --show-unset`).
- **One ambiguous constraint:** `compute.vmExternalIpAccess` returned an etag without a visible policy body — possibly inherited from a parent org/folder we don't have read access to. **Does not affect Cloud Run** (which uses a Google-managed HTTPS endpoint, not a VM external IP). **Flag for Aashir/Eric:** if their Compute Engine VM provisioning fails with "external IP denied by org policy", this is the likely cause — they'd need to either request the policy be relaxed or use Cloud NAT for outbound internet from a no-external-IP VM.
- **APIs enabled (all required services confirmed on):** `run.googleapis.com`, `artifactregistry.googleapis.com`, `cloudbuild.googleapis.com`, `compute.googleapis.com`, `storage.googleapis.com`, `monitoring.googleapis.com`, `logging.googleapis.com`.
- **Conclusion for the Container slice:** No NAT, Cloud Router, VPC connector, custom VPC, or `--ingress=internal` configuration needed. Cloud Run deploys with public ingress and default networking.

---

## Progress log

- ✅ **Phase 0 — Local setup:** gcloud configured to project `ancient-acumen-486002-j4`, Python venv with torch/torchvision/onnx/tritonclient
- ✅ **Phase 1 — Repo structure:** initial skeleton committed; `CLAUDE.md` lives at `cloudrun/CLAUDE.md`
- ✅ **Phase 2 — Model export:** `model/export_model.py` written and run successfully. Produced `resnet50.onnx` (~98 MB, single self-contained file via `dynamo=False`), verified structurally and numerically (PyTorch vs ONNX Runtime outputs match within FP tolerance), SHA-256 hash committed to `model/resnet50.onnx.sha256`
- ⬜ **Phase 3 — Triton container image:** Dockerfile + config.pbtxt
- ⬜ **Phase 4 — Push to Artifact Registry**
- ⬜ **Phase 5 — Deploy to Cloud Run**
- ⬜ **Phase 6 — Build harness**
- ⬜ **Phase 7+ — Run experiments, document, commit**

**Decision: skip the GCS bucket for the model.** The export is deterministic from `model/export_model.py` + the committed SHA-256 hash. Anyone needing the model runs `python model/export_model.py` and verifies via `shasum -a 256 -c model/resnet50.onnx.sha256`. Removes a redundant cloud-auth dependency for a one-time fixed artifact.

---

## Scope of this slice

Two responsibilities:

1. **Deploy Triton + ResNet-50 to Cloud Run** as the "container" configuration in the four-way comparison.
2. **Build and maintain the shared benchmarking harness** that all four teammates run against their own deployments.

Aggregation across configs is deferred to the end of the project — focus here is just on getting the Cloud Run deployment up and the harness distributed.

---

## What Cloud Run is (and isn't)

Cloud Run is GCP's managed container platform: hand it an image, get an HTTPS endpoint. It runs containers in a sandbox, autoscales them based on traffic, and bills per 100ms of request handling time.

Sits between an always-on VM (like Aashir's GPU box) and a true serverless function (like Ethan's Cloud Function). That hybrid nature is what we're measuring.

**Hardware:** CPU-only for our deployment. No GPU. ResNet-50 inference runs on the 4 vCPUs allocated to the container.

---

## The three things that will bite us if ignored

1. **Autoscaling adds variance.** Default behavior spins up new instances under load — a request hitting a cold-starting second instance can take 5–30s. Fix with `--min-instances=1 --max-instances=1`.

2. **Cloud Run concurrency ≠ harness concurrency.** Cloud Run's `--concurrency` flag controls how many requests one instance handles in parallel before queueing. Set this ≥ 100 so the harness's max concurrency level reaches Triton without queueing at the Cloud Run frontend.

3. **CPU throttling is on by default.** Cloud Run throttles CPU between requests unless we pass `--no-cpu-throttling`. Without this flag, CPU utilization measurements are unreliable.

---

## Required deployment flags

| Flag | Value | Why |
|---|---|---|
| `--min-instances` | 1 | Keep one instance warm; no scale-to-zero |
| `--max-instances` | 1 | Disable autoscaling |
| `--concurrency` | 100 | Allow all harness requests to reach Triton in parallel |
| `--cpu` | 4 | Per protocol |
| `--memory` | 8Gi | Per protocol; Triton + ResNet-50 needs the room |
| `--port` | 8000 | Triton HTTP port |
| `--timeout` | 60 | Per-request timeout (protocol uses 30s; headroom) |
| `--no-cpu-throttling` | (flag) | Full CPU always; required for clean utilization metrics |
| `--region` | us-central1 | Per protocol |
| `--allow-unauthenticated` | (flag) | Easier for harness; lock down later if needed |

---

## Triton container image

**Base image:** `nvcr.io/nvidia/tritonserver:24.01-py3` (decided — pinned for the whole team).

**Model export details (from `model/export_model.py`):**
- Input tensor name: `input`, shape `[1, 3, 224, 224]`, dtype FP32
- Output tensor name: `output`, shape `[1, 1000]`, dtype FP32
- ONNX opset version: **18** (the legacy TorchScript exporter handles opset 18 cleanly; opset 17 attempt failed with downgrade conversion error in newer torch/onnx versions)
- Fixed shape (no dynamic axes)
- Single-file export forced via `dynamo=False` (avoids external `.onnx.data` weight files)
- Pretrained weights: `ResNet50_Weights.IMAGENET1K_V2`

These names and shapes must match what `config.pbtxt` declares verbatim, or Triton fails to load the model.

**Model repository structure Triton expects:**

```
model_repository/
└── resnet50/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

**Decision: bake the model into the image.** Reasons:
- Eliminates fetch time as a variance source on cold start
- Matches what the VM-based deployments are doing more closely
- Model is fixed for the duration of the study

**`config.pbtxt` strategy:** single file with `instance_group { kind: KIND_AUTO }` so Triton picks the right backend per environment without us maintaining two configs (decided — Aashir's GPU and your Cloud Run CPU can use the same file).

---

## Build and deploy steps

1. Run `python model/export_model.py` from repo root to produce `resnet50.onnx` locally
2. Copy it into the model repo: `cp resnet50.onnx cloudrun/model_repository/resnet50/1/model.onnx`
3. Write `Dockerfile` — `FROM nvcr.io/nvidia/tritonserver:24.01-py3`, `COPY model_repository /models`, expose 8000
4. Build image locally: `cd cloudrun && docker build -t triton-resnet50:v1 .`
5. Smoke-test locally: `docker run -p 8000:8000` + `curl http://localhost:8000/v2/health/ready`
6. Push to Artifact Registry: `us-central1-docker.pkg.dev/ancient-acumen-486002-j4/benchmark-images/triton-resnet50:v1`
7. `gcloud run deploy` with the flags table above
8. Smoke-test the live endpoint with a single inference request before handing the URL to the harness

---

## Performance expectations

ResNet-50, FP32, batch size 1, on 4 vCPUs via ONNX runtime:
- Per-inference latency: roughly **80–150ms**
- Compare to ~5–10ms on Aashir's T4 GPU

The Cloud Run story isn't raw speed — it's **cost efficiency, deployment simplicity, and concurrency scaling behavior**. Frame results that way in the final report.

---

## What the Cloud Run config compares against

| Comparison | What the delta tells us |
|---|---|
| Cloud Run vs. Eric's standard CPU VM | Cost of managed containers + gVisor sandboxing vs. raw VM |
| Cloud Run vs. Ethan's Cloud Functions | Long-lived container model vs. function-style serverless |
| Cloud Run vs. Aashir's GPU VM | Cost-per-1000-inferences tradeoff (slower per request, cheaper when idle) |

---

## Benchmarking harness (shared across all 4 teammates)

**This is a Python program, not a bash script.** A thin bash wrapper (`run_all.sh`) loops over concurrency levels and run numbers, but the real benchmarking logic lives in Python.

### Why Python, not bash

The harness needs to: spawn N concurrent HTTP clients, send tensors as the inference payload, time each request with sub-millisecond precision, collect per-request data into a structured CSV, and handle Triton-specific protocol details. Bash can't do any of that cleanly. Python with `tritonclient` is the right tool.

### Architecture (layer cake)

| Layer | Tool | Job |
|---|---|---|
| `run_all.sh` | bash | Loop over (concurrency × run number), invoke Python harness 20 times |
| `harness.py` | Python | For one (config, concurrency, run): spawn N concurrent clients, send 200 requests each, write CSV |
| Concurrent clients | Python `asyncio` or `concurrent.futures` | Each client sends 200 sequential HTTP requests, times each |
| Triton server | (remote) | Runs ResNet-50 inference, returns prediction |

### Required library

Use **`tritonclient`** (NVIDIA's official Triton Python client) instead of hand-rolling HTTP requests. Handles tensor serialization, the right endpoint paths (`/v2/models/resnet50/infer`), and gRPC if we ever switch from HTTP. Install: `pip install tritonclient[http]`.

```python
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url=endpoint)
inputs = [httpclient.InferInput("input", [1, 3, 224, 224], "FP32")]
inputs[0].set_data_from_numpy(image_array)

start = time.perf_counter()
response = client.infer("resnet50", inputs)
latency_ms = (time.perf_counter() - start) * 1000
```

### Reusable scaffolding from Project 1 benchmarking script

The Project 1 performance-modeling script (`benchmark.py` from prior coursework) is **in-process and training-focused** — its core loop doesn't transfer. But these patterns are worth lifting:

- `get_hardware_info()` — reuse as-is for the per-config notes file
- argparse skeleton with `--env-name` style tagging — adapt for `--config-name`, `--concurrency`, `--run`
- JSON-structured metadata output — same shape for the notes file
- Warmup-discard logic — same pattern, applied to first 20 requests instead of first 5 batches
- ImageNet normalization constants (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) — same values, applied client-side before sending the tensor

What to **drop** from Project 1: the training loop, CUDA event timing, FLOPs/roofline metrics, the multi-model loop, the `thop` dependency.

### Harness CLI contract

```bash
python harness.py \
  --endpoint <url-or-ip:port> \
  --config-name <gpu|cpu|confidential|cloudrun|cloudfunction> \
  --concurrency <1|10|50|100> \
  --run <1..5> \
  --requests 200 \
  --warmup 20 \
  --output results/
```

Per the protocol, this writes one CSV named `results_{config-name}_{concurrency}_{run}.csv`.

### Per-request CSV schema

One row per request:

```
request_id, client_id, send_ts, receive_ts, latency_ms, status_code, success
```

Aggregation (p50/p95/p99, throughput, error rate) happens later, at the end of the project, across everyone's CSVs. The harness does **not** compute percentiles inline — raw data only.

### Metrics summary

The harness collects the data; aggregation produces the report-ready numbers.

| Metric | Collected by | Computed when |
|---|---|---|
| Per-request latency (ms) | Harness, client-side end-to-end | Live, per CSV row |
| p50 / p95 / p99 latency | — | Aggregation phase, from latency column |
| Throughput (req/s) | Harness records run start/end timestamps | Aggregation phase: total successes ÷ duration |
| Error rate | Harness records `success` column | Aggregation phase: failures ÷ total |
| Cold start (ms) | Harness `--cold-start` flag — first request after deploy, pre-warmup | Per-config one-time measurement (Cloud Functions especially; Cloud Run with min-instances=1 has minimal cold start) |
| CPU/GPU/RAM utilization | Cloud Monitoring, sampled every 5s during run | Pulled post-hoc from Cloud Monitoring API into the notes file |
| Cost per 1000 inferences | — | Aggregation phase: utilization × prices recorded in notes file |

---

## Outputs to produce

**Cloud Run deployment artifacts (Shopnil's slice):**
- `Dockerfile` and `model_repository/` in the shared repo
- `deploy_cloudrun.sh` script (write fresh, matching the flags table above)
- Cloud Run endpoint URL (share with team for reference; only Shopnil benchmarks against it)

**Shared harness (Shopnil owns, all teammates use):**
- `harness.py` — Python program using `tritonclient`, implements the CLI contract above
- `run_all.sh` — thin bash wrapper that invokes `harness.py` 20 times (4 concurrency × 5 runs)
- `requirements.txt` — pinned dependencies (`tritonclient[http]`, `numpy`, `Pillow`)
- README snippet showing how teammates run it against their own endpoints

**Experimental record (per teammate):**
- Raw harness output: `results_cloudrun_{1,10,50,100}_{1..5}.csv` in `/results/` — 20 CSVs with per-request rows
- Notes file (JSON): hardware info from `get_hardware_info()`, instance config, Triton version pinned tag, Docker image SHA, run timestamps, GCP prices used at run time, any deviations from protocol

---

## Open items to confirm with the team before deploying

- [x] ~~Triton version tag~~ → **decided: `nvcr.io/nvidia/tritonserver:24.01-py3`**
- [x] ~~`config.pbtxt` strategy~~ → **decided: single file with `KIND_AUTO`**
- [x] ~~Artifact Registry repo name~~ → **decided: `benchmark-images` in `us-central1`**
- [x] ~~ONNX opset version~~ → **decided: 18 (single-file export via `dynamo=False`)**
- [x] ~~Model distribution method~~ → **decided: skip GCS bucket; teammates run the export script locally and verify via SHA-256**
- [ ] Whether the harness endpoint should use authenticated invocation (probably no for simplicity)
- [ ] Aashir/Eric: confirm `compute.vmExternalIpAccess` doesn't block their VM creation before they spend time on deployment scripts

---

## Canonical model artifact

- **Source script:** `model/export_model.py` (committed) — uses `dynamo=False` and opset 18 to produce a single self-contained .onnx file
- **Hash file:** `model/resnet50.onnx.sha256` (committed) — for integrity verification
- **Binary:** `resnet50.onnx` (~98 MB, single file) — **NOT committed**, gitignored. Lives in:
  - Local filesystem after running the export script
  - Built into Docker image at `/models/resnet50/1/model.onnx` during Phase 3
- **Reproducibility:** any teammate runs `python model/export_model.py` and verifies with `shasum -a 256 -c model/resnet50.onnx.sha256`. No cloud bucket needed.
