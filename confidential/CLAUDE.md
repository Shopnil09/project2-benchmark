# CLAUDE.md — Confidential Computing Slice (CPU baseline + AMD SEV)

**Owner:** Yang-Jung (Eric) Chen
**Project:** DNN Inference Benchmarking on GCP — Confidential Computing slice
**Model:** ResNet-50 (ONNX, FP32, batch size 1)
**Serving framework:** Triton Inference Server (`nvcr.io/nvidia/tritonserver:24.01-py3`)
**Region/zone:** us-central1 / us-central1-a
**GCP project:** `applied-ml-cloud` (Eric's personal project, not the team project)

---

## What this slice owns

Two GCE VMs deployed as a *matched pair*:

| VM name | Machine type | Confidential compute | Purpose |
|---|---|---|---|
| `triton-cpu-standard` | n2d-standard-4 | (off) | The team's CPU baseline (`config-name: cpu` in the harness) |
| `triton-cpu-confidential` | n2d-standard-4 | AMD SEV | TEE variant (`config-name: confidential`) |

Same machine type, same image (Ubuntu 22.04 LTS), same Triton image, byte-identical model. Only the host firmware/kernel feature differs. The delta in latency, throughput, and cost between these two is the **TEE overhead** that becomes a focused subsection in the team's final report.

Because Aashir owns GPU and Shopnil/Ethan own managed services, the standard-CPU VM also doubles as **the team's only IaaS CPU baseline** in the four-way deployment comparison.

---

## Key correction to the team protocol

The original protocol (see `convo.md`) specified `n2-standard-4` for both VMs. **That was wrong**: AMD SEV is only available on AMD EPYC machine series, which on GCP means `n2d-*`, `c2d-*`, or `c3d-*`. The `n2-*` series is Intel and cannot enable AMD SEV.

**This slice uses `n2d-standard-4`** (4 vCPU, 16 GB RAM, AMD EPYC) for both VMs. This keeps the matched-pair comparison clean — the only varying factor between the two VMs is the SEV flag.

This deviation from the written protocol must be flagged in the final report's methodology section.

---

## Why Ubuntu 22.04, not Container-Optimized OS

Shopnil's Cloud Run slice uses COS, but COS does not currently support `--confidential-compute=SEV`. To keep the standard and confidential VMs *truly* matched, both run Ubuntu 22.04 LTS. Triton runs in a container on both regardless, so the OS difference vs. the Cloud Run slice does not affect the inference path — the model still runs inside the same `nvcr.io/nvidia/tritonserver:24.01-py3` image.

---

## Networking constraint — `compute.vmExternalIpAccess` is enforced

The `applied-ml-cloud` project has the `compute.vmExternalIpAccess` org policy set (verified — the constraint has a non-empty etag, inherited from a parent org we can't read). **No VM in this project can be assigned an external IP.** This shapes the architecture:

- All three VMs (standard, confidential, runner) are deployed with `--no-address`.
- SSH happens via **IAP TCP forwarding** (`gcloud compute ssh --tunnel-through-iap`), which doesn't need an external IP.
- The benchmarking harness runs on a **third VM in the same VPC** (the "harness runner") that talks to Triton over the default VPC's internal IPs.

**Why a runner VM and not IAP-tunnel-from-laptop**: IAP TCP forwarding adds 5–15 ms per request and multiplexes "concurrent clients" through Google's edge proxy, which would distort latency *and* the concurrency contract the team protocol depends on. An in-VPC runner gives clean internal-IP routing with no proxy in the path.

---

## What's in this directory

| File | Purpose |
|---|---|
| [deploy_vm.sh](deploy_vm.sh) | Provisions one Triton VM (`standard` or `confidential` variant) with `--no-address`. Idempotent. Sets `--confidential-compute-type=SEV` and `--maintenance-policy=TERMINATE` on the confidential variant. |
| [setup_vm.sh](setup_vm.sh) | GCE startup script for the Triton VMs. Installs Docker, exports the canonical ResNet-50 ONNX model (opset 18, dynamo=False), lays out the Triton model repository, and starts Triton on TCP 8000. Identical for both variants. |
| [deploy_runner.sh](deploy_runner.sh) | Provisions the in-VPC harness-runner VM (e2-standard-2). Inline startup script installs Python + `tritonclient[http]`. Eric SSHs in via IAP and uploads `harness/` via `gcloud compute scp`. |
| [firewall.sh](firewall.sh) | Creates two rules in the default VPC: `allow-iap-ssh` (35.235.240.0/20 → tcp:22) and `allow-triton-internal` (10.0.0.0/8 → tcp:8000). Both target the `triton-server` tag. |
| [teardown.sh](teardown.sh) | Deletes the three VMs and the slice's firewall rules. |
| [README.md](README.md) | Deliverable: explains the slice in plain English — role, what was deployed, what was run, expected and actual results. |

---

## Build and deploy procedure

All commands run from `project2-benchmark/` on your laptop. Authenticated as `yc4670@columbia.edu` against project `applied-ml-cloud` (active gcloud config: `amlc`).

```bash
# 0. Activate gcloud config (one-time)
gcloud config configurations activate amlc
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a

# 1. Enable APIs (one-time)
gcloud services enable compute.googleapis.com \
                       monitoring.googleapis.com \
                       logging.googleapis.com

# 2. Open firewall (IAP SSH + internal Triton on 8000)
bash confidential/firewall.sh

# 3. Provision the runner VM and Triton VMs (run all three concurrently if you want)
bash confidential/deploy_runner.sh applied-ml-cloud
bash confidential/deploy_vm.sh     applied-ml-cloud standard
bash confidential/deploy_vm.sh     applied-ml-cloud confidential

# 4. Capture the internal IPs (Triton VMs have NO external IP)
STD_IP=$(gcloud compute instances describe triton-cpu-standard \
  --zone us-central1-a --format='get(networkInterfaces[0].networkIP)')
CONF_IP=$(gcloud compute instances describe triton-cpu-confidential \
  --zone us-central1-a --format='get(networkInterfaces[0].networkIP)')
echo "Standard:     $STD_IP"
echo "Confidential: $CONF_IP"

# 5. Wait ~5-7 min for both Triton VMs' startup scripts to finish.
#    Watch progress on the standard VM:
gcloud compute instances get-serial-port-output triton-cpu-standard \
  --zone us-central1-a | tail -50

# 6. Push the harness code to the runner VM
gcloud compute scp --tunnel-through-iap --zone us-central1-a --recurse \
  harness/ harness-runner:~

# 7. SSH into the runner and smoke-test from inside the VPC
gcloud compute ssh harness-runner --zone us-central1-a --tunnel-through-iap
# from inside the runner:
curl -fsS http://${STD_IP}:8000/v2/health/ready  && echo "standard OK"
curl -fsS http://${CONF_IP}:8000/v2/health/ready && echo "confidential OK"

# 8. Run the benchmarks from inside the runner VM
bash harness/run_all.sh http://${STD_IP}:8000  cpu          # ~45-75 min
bash harness/run_all.sh http://${CONF_IP}:8000 confidential # ~45-75 min

# 9. From your laptop, pull the CSVs back
gcloud compute scp --tunnel-through-iap --zone us-central1-a --recurse \
  harness-runner:~/results ./

# 10. Tear down (deletes all three VMs + firewall rules)
bash confidential/teardown.sh applied-ml-cloud
```

---

## The three things that will bite if ignored

1. **Confidential VMs cannot live-migrate.** That's why `--maintenance-policy=TERMINATE` is required on the confidential variant. Without it, `gcloud` rejects the create call.
2. **Same-day runs.** Both VMs must be benchmarked on the same calendar day, ideally back-to-back during off-peak hours, so noisy-neighbor variance affects both equally and one set of GCP prices applies to both.
3. **Model integrity.** `setup_vm.sh` runs the export inline on each VM — the SHA-256 of the produced `model.onnx` must match `model/resnet50.onnx.sha256` in the team repo. If the export drifts (newer torch/torchvision can produce different bit-exact ONNX), the comparison is invalidated.

---

## Triton flags on the VMs

The `docker run` invocation in `setup_vm.sh` mirrors Shopnil's Cloud Run Dockerfile:

```
tritonserver \
  --model-repository=/models \
  --strict-model-config=true \
  --allow-http=true \
  --allow-metrics=true \
  --log-verbose=0
```

- `--strict-model-config=true` — fails fast if `config.pbtxt` doesn't match the ONNX tensor names/shapes. This is the bug Shopnil hit during Cloud Run debugging; surfacing it on VM startup is the same pattern.
- `--max-batch-size=0` (set in `config.pbtxt`) — the harness sends batch-1 requests with shape `[1, 3, 224, 224]` already including the batch dim.

---

## Performance expectations (predictions to verify)

ResNet-50, FP32, batch 1, on n2d-standard-4 (4 vCPU AMD EPYC) via ONNX Runtime CPU backend:

- **Standard VM**: per-request p50 latency around **80–150 ms** at concurrency 1; throughput **~7–12 req/s** at concurrency 1, scaling sublinearly to **~30–50 req/s** at concurrency 100 (4 vCPUs cap throughput).
- **Confidential VM**: **5–15% latency overhead** vs. standard, consistent with published AMD SEV overhead figures for memory-bound CPU workloads. Throughput drop proportional.
- **Cost delta**: AMD SEV pricing premium is **~6%** on n2d. Combined with the throughput drop, expect cost-per-1k-inferences to be **~11–22% higher** on the confidential VM.

These are predictions — replace with measured medians in `README.md` after the runs complete.

---

## What the deltas will tell us

| Comparison | What the delta tells us |
|---|---|
| Standard CPU VM vs. Aashir's GPU VM | Per-request inference cost of CPU vs. GPU; where each wins on the cost-per-1k axis |
| Standard CPU VM vs. Shopnil's Cloud Run | Cost of managed-container abstraction (gVisor sandbox, Cloud Run frontend) vs. raw VM |
| Standard CPU VM vs. Ethan's Cloud Functions | IaaS vs. FaaS at steady state (and cold-start delta from Ethan's measurement) |
| **Standard CPU vs. Confidential CPU** (this slice's headline) | The TEE overhead — the runtime cost of memory-encrypted execution for CPU inference |

---

## Risks tracked

- **External IP org policy** *(confirmed in effect)*: `compute.vmExternalIpAccess` blocks external IPs at the Columbia GCP-org level. Worked around by deploying all VMs with `--no-address`, opening IAP SSH, and running the harness from a runner VM in the same VPC. **No internet egress** from these VMs without Cloud NAT — that's why `setup_vm.sh` installs Python deps via apt + pip from PyPI and pulls the Triton image from NGC: those reach the public internet via Google's *Private Google Access* on the default VPC. If a future change adds a dependency hosted outside the Google-private-network whitelist, a Cloud NAT gateway would be needed.
- **Confidential VM zonal capacity**: if SEV-capable n2d capacity is unavailable in `us-central1-a`, fall back to `us-central1-b` or `-c` and run **both** VMs in the new zone (do not split zones — that introduces variance).
- **Triton image pull**: `nvcr.io/nvidia/tritonserver:24.01-py3` reaches NGC via Private Google Access on the default VPC. If the pull fails, this is the most likely diagnosis; check `gcloud compute networks subnets describe default --region=us-central1 --format='get(privateIpGoogleAccess)'` returns `True`.
- **Billing**: ~$0.135/hr (standard) + ~$0.143/hr (confidential, ~6% SEV premium) + ~$0.067/hr (e2-standard-2 runner). For ~3 hours of benchmarking + ~1 hour of debugging, expect total spend under $2.

---

## Outputs to commit

After both 20-CSV sets land in `results/cpu/` and `results/confidential/`:

- `results/cpu/results_cpu_{1,10,50,100}_{1..5}.csv` — 20 files
- `results/confidential/results_confidential_{1,10,50,100}_{1..5}.csv` — 20 files
- `results/cpu/notes.json` and `results/confidential/notes.json` — machine type, image family, kernel version (`uname -r` from inside the VM), Docker version, Triton image SHA, run timestamps, GCP on-demand prices used, and a "deviations" field (record the n2d-vs-n2 correction).
- `confidential/README.md` — actual-results table filled in with measured medians and the standard-vs-confidential deltas.
