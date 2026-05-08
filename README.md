# Project 2 — DNN Inference Benchmarking on GCP

Performance benchmarking of ResNet-50 inference across five GCP deployment configurations: GPU VM, CPU VM, Confidential VM, Cloud Run, and Cloud Functions. Includes a confidential-computing overhead study (AMD SEV) and an interactive results dashboard.

**Team:** Aashir Khan, Ethan Chang, Yang-Jung (Eric) Chen, Shopnil Shahriar
**Model:** ResNet-50, ONNX, FP32, batch size 1
**Serving framework:** Triton Inference Server (`nvcr.io/nvidia/tritonserver:24.01-py3`); `onnxruntime` + a Triton-v2 HTTP shim for Cloud Functions
**Region:** us-central1
**Project:** `ancient-acumen-486002-j4` (Cloud Run, Cloud Functions, GPU); `applied-ml-cloud` (Confidential)

**Dashboard:** https://shopnil09.github.io/project2-benchmark/

---

## Repository layout

```
project2-benchmark/
├── index.html              # Interactive results dashboard (Chart.js)
├── model/                  # ResNet-50 → ONNX export (canonical artifact)
├── harness/                # Shared benchmarking harness (used by all configs)
├── cloudrun/               # Cloud Run deployment (Triton in a container)
├── cloudfunction/          # Cloud Functions 2nd gen deployment (onnxruntime + Triton-v2 shim)
├── confidential/           # CPU VM + Confidential VM deployments (AMD SEV study)
└── results/
    ├── cloudrun/from_gce/  # 20 CSVs (Shopnil Shahriar, run from in-region GCE client)
    ├── cloudfunction/      # 20 CSVs + NOTES.md (Ethan Chang)
    └── gpu/                # 20 CSVs + notes_gpu.json (Aashir Khan)
confidential/results/
    ├── cpu/                # 20 CSVs (Yang-Jung Chen — standard n2d-standard-4)
    └── confidential/       # 20 CSVs (Yang-Jung Chen — n2d-standard-4 + AMD SEV)
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
| Cloud Run | Shopnil Shahriar | 4 vCPU / 8 GB (managed) | 20/20 | Benchmarked from in-region GCE client (no external IPs allowed) |
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

Single source of truth for the model used by every config. Loads `ResNet50_Weights.IMAGENET1K_V2`, exports to ONNX opset 18 via the legacy TorchScript exporter (`dynamo=False`) so the result is a **single self-contained file** rather than `.onnx + .onnx.data`. Validates structure (`onnx.checker`) and numerics (PyTorch vs ONNX Runtime within 1e-4) before writing the SHA-256 hash.

| File | Purpose |
|---|---|
| [model/export_model.py](model/export_model.py) | Exports ResNet-50 → ONNX, verifies, writes SHA-256. |
| [model/resnet50.onnx.sha256](model/resnet50.onnx.sha256) | Committed integrity hash — every teammate verified the same artifact. |

```bash
python model/export_model.py
shasum -a 256 -c model/resnet50.onnx.sha256
```

**Tensor contract:** input `[1, 3, 224, 224]` FP32 → output `[1, 1000]` FP32. Input/output names: `input` / `output`.

---

## [harness/](harness/) — Shared benchmarking harness

The single tool every teammate runs against their own deployment. Spawns N concurrent threads with `ThreadPoolExecutor`, each holding its own `tritonclient.http.InferenceServerClient`, sending sequential inference requests and recording per-request latency. Cloud Functions also speaks Triton's KServe v2 protocol on the wire, so the harness targets it unmodified.

| File | Purpose |
|---|---|
| [harness/harness.py](harness/harness.py) | One (config, concurrency, run): probes server readiness, builds the ImageNet-normalized FP32 input, spawns N threads × 200 sequential requests, drops the first 20 per thread, writes one CSV. |
| [harness/run_all.sh](harness/run_all.sh) | Loops concurrency ∈ {1, 10, 50, 100} × run ∈ {1..5} = 20 invocations. 5s pause between runs; reports failed runs at the end. |
| [harness/requirements.txt](harness/requirements.txt) | `tritonclient[http]==2.43.0`, `numpy>=1.24`, `Pillow>=10`. |

```bash
pip install -r harness/requirements.txt
bash harness/run_all.sh <endpoint-url> <config-name>
# config-name ∈ {gpu, cpu, confidential, cloudrun, cloudfunction}
```

**CSV schema:** `request_id, client_id, send_ts, receive_ts, latency_ms, success, error` (one row per measured request, 180 rows × 20 CSVs per config).

---

## Running the harness inside the VPC (no-external-IP environments)

The `applied-ml-cloud` project (used for the confidential slice) enforces `compute.vmExternalIpAccess`, so no VM can have a public IP. Even where external IPs are allowed, IAP TCP forwarding from a laptop adds 5–15 ms per request and multiplexes "concurrent clients" through Google's edge proxy — that distorts both latency and the concurrency contract the harness depends on. The fix is to run the harness from a small **runner VM in the same VPC** that talks to Triton over internal IPs.

### Architecture

```
[laptop] ── IAP SSH ──> [harness-runner]  ── internal IP ──> [triton-cpu-standard]
                        (no external IP)                     (no external IP)
                              │                              [triton-cpu-confidential]
                              │
                              └── Cloud NAT ──> apt / PyPI / NGC
```

### One-time infrastructure (all idempotent)

| Step | Script | What it does |
|---|---|---|
| Firewall | [confidential/firewall.sh](confidential/firewall.sh) | `allow-iap-ssh` (IAP range `35.235.240.0/20` → `tcp:22`) + `allow-triton-internal` (`10.0.0.0/8` → `tcp:8000`) on tag `triton-server`. |
| Egress | [confidential/cloud_nat.sh](confidential/cloud_nat.sh) | Cloud Router (`nat-router`) + Cloud NAT (`nat-config`) in us-central1 so no-external-IP VMs can reach apt mirrors, PyPI, and `nvcr.io`. |

```bash
bash confidential/firewall.sh  [project-id]
bash confidential/cloud_nat.sh [project-id]
```

### Provision the runner VM

[confidential/deploy_runner.sh](confidential/deploy_runner.sh) creates `harness-runner` (`e2-standard-2`, us-central1-a, no external IP, tag `triton-server`). Its inline GCE startup script installs Python and the harness's runtime deps (`tritonclient[http]==2.43.0`, `numpy`, `Pillow`). After ~2 minutes the runner is ready.

```bash
bash confidential/deploy_runner.sh <project-id>

# Watch the startup script finish without SSHing in:
gcloud compute instances get-serial-port-output harness-runner \
  --zone us-central1-a | grep "runner ready"
```

### Push harness, run benchmark, pull results

```bash
# 1. Upload harness/ from your laptop to the runner via IAP
gcloud compute scp --tunnel-through-iap --zone us-central1-a --recurse \
  harness/ harness-runner:~

# 2. Capture the Triton VM's internal IP (no external IP per org policy)
TRITON_IP=$(gcloud compute instances describe <triton-vm-name> \
  --zone us-central1-a --format='get(networkInterfaces[0].networkIP)')

# 3. Smoke-test reachability from inside the VPC
gcloud compute ssh harness-runner --zone us-central1-a --tunnel-through-iap \
  --command="curl -fsS http://${TRITON_IP}:8000/v2/health/ready && echo OK"

# 4. Run the full 20-CSV protocol from inside the VPC.
#    Either keep the SSH session open and run interactively...
gcloud compute ssh harness-runner --zone us-central1-a --tunnel-through-iap
# inside the runner:
bash harness/run_all.sh http://${TRITON_IP}:8000 <config-name>
# config-name ∈ {gpu, cpu, confidential, cloudrun, cloudfunction}

#    ...or fire-and-forget via a single SSH command:
gcloud compute ssh harness-runner --zone us-central1-a --tunnel-through-iap \
  --command="bash harness/run_all.sh http://${TRITON_IP}:8000 <config-name>"

# 5. Pull the 20 CSVs back to the laptop
gcloud compute scp --tunnel-through-iap --zone us-central1-a --recurse \
  harness-runner:~/results/<config-name> ./results/
```

### Teardown

[confidential/teardown.sh](confidential/teardown.sh) deletes both Triton VMs, the runner, the firewall rules, the Cloud NAT, and the Cloud Router. NAT bills hourly even when idle, so don't leave it up after the experiment finishes.

```bash
bash confidential/teardown.sh <project-id>
# or keep firewall + NAT for follow-on runs:
bash confidential/teardown.sh <project-id> --keep-firewall
```

### Notes & gotchas

- All three VMs (Triton standard, Triton confidential, runner) need the `triton-server` network tag for the firewall rules to match. `deploy_vm.sh` and `deploy_runner.sh` already set this.
- The runner needs `--scopes=cloud-platform` (already set) so `gcloud` from inside the VM can describe sibling instances if you script multi-stage runs.
- `e2-standard-2` (2 vCPU / 8 GB) is enough — the harness's 100 concurrent threads + `tritonclient[http]` event loop sit well below that ceiling. Bumping the runner won't change measurements.
- Cloud Run benchmarking uses the same in-VPC pattern but inlined into one script — [cloudrun/benchmark_from_gce.sh](cloudrun/benchmark_from_gce.sh) provisions an ephemeral runner, runs the protocol, pulls the CSVs, and deletes the runner in a single command.
- Don't run two `run_all.sh` invocations in parallel against the same runner — both processes would compete for the runner's CPU and skew the latency numbers. Run `cpu` first, then `confidential` (or vice versa) sequentially.

---

## [cloudrun/](cloudrun/) — Cloud Run (Shopnil Shahriar)

Triton Inference Server in a Docker container deployed to Cloud Run with the model baked into the image. `--min-instances=1 --max-instances=1 --concurrency=100 --no-cpu-throttling` to remove autoscaling and CPU-throttling variance. Benchmarking runs from a **no-external-IP GCE VM in us-central1** (org policy blocks external IPs and laptop-to-Cloud-Run RTT would dominate the measurement); the helper script provisions that VM via IAP + Cloud NAT, runs the harness, scps the CSVs back, and tears the VM down.

| File | Purpose |
|---|---|
| [cloudrun/Dockerfile](cloudrun/Dockerfile) | `FROM nvcr.io/nvidia/tritonserver:24.01-py3`, copies `model_repository/` into `/models`, exposes 8000/8001/8002. |
| [cloudrun/model_repository/resnet50/config.pbtxt](cloudrun/model_repository/resnet50/config.pbtxt) | Triton model config — FP32, `max_batch_size: 0`, `KIND_AUTO`. |
| [cloudrun/benchmark_from_gce.sh](cloudrun/benchmark_from_gce.sh) | Provisions an in-region n2/e2 client VM (no external IP, IAP SSH, Cloud NAT egress), uploads the harness, runs the full 20-CSV protocol, pulls CSVs into `results/cloudrun/from_gce/`, deletes the VM. |
| [cloudrun/test_inference.py](cloudrun/test_inference.py) | Smoke test — single synthetic request, prints top-5 + latency. |
| [cloudrun/CLAUDE.md](cloudrun/CLAUDE.md) | Full deployment notes and pitfalls. |

```bash
# 1. Build the model artifact and stage it inside the image
python model/export_model.py
cp resnet50.onnx cloudrun/model_repository/resnet50/1/model.onnx

# 2. Build, push, deploy (see cloudrun/CLAUDE.md for the full gcloud run deploy command)
cd cloudrun && docker build -t triton-resnet50:v1 .
# ...push to Artifact Registry, then `gcloud run deploy ...`

# 3. Smoke-test the live endpoint
python cloudrun/test_inference.py --endpoint <cloud-run-url>

# 4. Run the benchmark from an in-region VM client
bash cloudrun/benchmark_from_gce.sh <cloud-run-url>
```

---

## [cloudfunction/](cloudfunction/) — Cloud Functions 2nd gen (Ethan Chang)

Cloud Functions 2nd gen can't run a custom Docker image, so Triton is replaced with a tiny in-process **`onnxruntime`-backed adapter that speaks Triton's KServe v2 HTTP protocol on the wire** — the shared harness targets this endpoint indistinguishably from a real Triton server. `min-instances=1 max-instances=1 --concurrency=100` to match the Cloud Run sizing.

| File | Purpose |
|---|---|
| [cloudfunction/main.py](cloudfunction/main.py) | HTTP entry point (`triton_handler`). Loads the ONNX model at cold start, implements `/v2/health/{live,ready}`, `/v2/models/resnet50`, `/v2/models/resnet50/infer` — handles both Triton's binary-mixed and JSON-only tensor encodings. |
| [cloudfunction/deploy.sh](cloudfunction/deploy.sh) | `gcloud functions deploy --gen2 --cpu=4 --memory=8Gi --timeout=540s --min-instances=1 --max-instances=1 --concurrency=100`. Fetches the live URL via the REST API to sidestep a macOS protobuf bug in `gcloud functions describe`. |
| [cloudfunction/test_inference.py](cloudfunction/test_inference.py) | Smoke test, also reports first-request latency separately for cold-start sampling. |
| [cloudfunction/.gcloudignore](cloudfunction/.gcloudignore) | **Required** — without it, gcloud falls back to `.gitignore` (which excludes `*.onnx`) and the deploy archive ships without the model. |
| [cloudfunction/CLAUDE.md](cloudfunction/CLAUDE.md) | Deployment notes, cold-start methodology, pitfalls. |

```bash
# 1. Stage the model into the function source dir
python model/export_model.py
cp resnet50.onnx cloudfunction/

# 2. Deploy (prints the function URL on completion)
bash cloudfunction/deploy.sh

# 3. Smoke test + cold-start sample
python cloudfunction/test_inference.py --endpoint <function-url>

# 4. Run the benchmark
bash harness/run_all.sh <function-url> cloudfunction
```

**C=100 wall:** five C=100 runs each saw ~46% error rate (per-request 30s timeout firing on queued requests) at ~18 req/s steady-state — a structural ceiling on 4 vCPUs, not transient noise. Logged in [results/cloudfunction/NOTES.md](results/cloudfunction/NOTES.md).

---

## [confidential/](confidential/) — CPU VM + Confidential VM (Yang-Jung Chen)

Matched-pair deployment of a standard and an AMD SEV-encrypted GCE VM on identical `n2d-standard-4` hardware (AMD EPYC). The only varying factor is the SEV flag, so the latency/throughput delta isolates **TEE overhead**. The standard VM also serves as the team's CPU baseline. Org policy on the `applied-ml-cloud` project blocks external IPs, so the harness runs on a third **in-VPC runner VM** that talks to the two Triton VMs over internal IPs (laptop-via-IAP would add 5–15 ms per request and distort concurrency).

> **Methodology correction:** the original protocol specified `n2-standard-4` (Intel). AMD SEV requires AMD EPYC (`n2d-*`/`c2d-*`/`c3d-*`), so both VMs use `n2d-standard-4` to keep the matched-pair valid.

| File | Purpose |
|---|---|
| [confidential/firewall.sh](confidential/firewall.sh) | Default-VPC firewall rules: IAP SSH (`35.235.240.0/20:22`) + internal Triton (`10.0.0.0/8:8000`). |
| [confidential/cloud_nat.sh](confidential/cloud_nat.sh) | Cloud Router + Cloud NAT in us-central1 so the no-external-IP VMs can pull the Triton image and pip packages. |
| [confidential/deploy_vm.sh](confidential/deploy_vm.sh) | Provisions one Triton VM (`standard` or `confidential` variant). Sets `--confidential-compute-type=SEV` and `--maintenance-policy=TERMINATE` on the SEV variant. Idempotent. |
| [confidential/setup_vm.sh](confidential/setup_vm.sh) | GCE startup script — installs Docker, exports the canonical ONNX inline (opset 18, `dynamo=False`), writes `config.pbtxt`, starts Triton on TCP 8000. Identical for both variants. |
| [confidential/deploy_runner.sh](confidential/deploy_runner.sh) | Provisions the in-VPC harness-runner VM (e2-standard-2). |
| [confidential/teardown.sh](confidential/teardown.sh) | Deletes the three VMs and the slice's firewall rules. |
| [confidential/README.md](confidential/README.md) | Slice deliverable — role, deployments, results. |
| [confidential/explanation.md](confidential/explanation.md) | Plain-language project overview (Mandarin). |
| [confidential/CLAUDE.md](confidential/CLAUDE.md) | Full deployment notes and risk register. |

```bash
# 0. Activate the personal GCP project (one-time)
gcloud config configurations activate amlc

# 1. Networking + VMs
bash confidential/firewall.sh
bash confidential/cloud_nat.sh                  # if not already configured
bash confidential/deploy_vm.sh     applied-ml-cloud standard
bash confidential/deploy_vm.sh     applied-ml-cloud confidential
bash confidential/deploy_runner.sh applied-ml-cloud
# ~5–7 min for the Triton startup scripts to finish

# 2. Capture the internal IPs (no external IP per org policy)
STD_IP=$(gcloud compute instances describe triton-cpu-standard \
  --zone us-central1-a --format='get(networkInterfaces[0].networkIP)')
CONF_IP=$(gcloud compute instances describe triton-cpu-confidential \
  --zone us-central1-a --format='get(networkInterfaces[0].networkIP)')

# 3. Run the harness from inside the VPC
gcloud compute scp --tunnel-through-iap --zone us-central1-a --recurse \
  harness/ harness-runner:~
gcloud compute ssh harness-runner --zone us-central1-a --tunnel-through-iap
# inside the runner:
bash harness/run_all.sh http://${STD_IP}:8000  cpu
bash harness/run_all.sh http://${CONF_IP}:8000 confidential

# 4. Pull CSVs back, then tear it all down
gcloud compute scp --tunnel-through-iap --zone us-central1-a --recurse \
  harness-runner:~/results ./
bash confidential/teardown.sh applied-ml-cloud
```

Measured TEE overhead: AMD SEV adds **~0.7% p50 latency at C=10** and a **~6% hourly price premium** — effectively free in performance terms for this workload.

---

## GPU VM (Aashir Khan)

NVIDIA L4 GPU on a `g2-standard-4` Compute Engine instance running Triton Inference Server with the ONNX backend. Provisioning + run notes captured in [results/gpu/notes_gpu.json](results/gpu/notes_gpu.json) (machine type, GPU model, image digest, GCP pricing snapshot, deviations).

**Deviations from protocol:**
- NVIDIA L4 used instead of T4 — T4 was exhausted across all us-central1 zones and tested US regions at provisioning time.
- `g2-standard-4` instead of `n1-standard-4` — required machine type for L4.
- Harness ran on the VM (`localhost:8000`) — org policy blocks external IPs, so the network-latency component is zero and all measured latency is pure inference + Triton overhead.

---

## [results/](results/) — Raw benchmark data

```
results/<config>/results_<config>_<concurrency>_<run>.csv
results/cloudrun/from_gce/results_cloudrun_<concurrency>_<run>.csv
confidential/results/<config>/results_<config>_<concurrency>_<run>.csv
```

20 CSVs per config (4 concurrency levels × 5 runs), 180 measured rows per CSV (200 requests − 20 warmup). Aggregated p50/p95/p99 and throughput are in [the dashboard](https://shopnil09.github.io/project2-benchmark/); raw per-request data in CSV is the canonical record. Per-config notes (`NOTES.md`, `notes_gpu.json`) record GCP pricing at run time, software versions, and any protocol deviations.

---

## [index.html](index.html) — Interactive dashboard

Self-contained Chart.js dashboard hosted via GitHub Pages at [shopnil09.github.io/project2-benchmark](https://shopnil09.github.io/project2-benchmark/). Reads the aggregated numbers inline (no backend) and renders latency-vs-concurrency, throughput, cost-per-1k, and the CPU-vs-Confidential overhead breakdown.

---

## End-to-end reproduction

```bash
# 1. Set up
python -m venv venv && source venv/bin/activate
pip install -r harness/requirements.txt

# 2. Build the model artifact
python model/export_model.py
shasum -a 256 -c model/resnet50.onnx.sha256

# 3. Deploy your config (see per-config CLAUDE.md / README.md)

# 4. Run the full benchmark protocol
bash harness/run_all.sh <your-endpoint> <config-name>
# config-name ∈ {gpu, cpu, confidential, cloudrun, cloudfunction}
```
