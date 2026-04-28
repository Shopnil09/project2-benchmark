# Confidential Computing Slice — Eric's Part

**Author:** Yang-Jung (Eric) Chen
**Project:** *Performance Benchmarking of DNN Inference Across Cloud Deployment Configurations with Confidential Computing Analysis*
**Course:** Columbia AMLC, Spring 2026

---

## My role

I own the **confidential-computing slice** of the team's GCP benchmarking study. Concretely, I deploy a *matched pair* of GCE virtual machines and run the team's shared benchmarking harness against both:

1. **`triton-cpu-standard`** — a standard `n2d-standard-4` VM. This is the team's **CPU baseline** in the four-way deployment comparison (alongside Aashir's GPU VM, Shopnil's Cloud Run container, and Ethan's Cloud Function).
2. **`triton-cpu-confidential`** — the same `n2d-standard-4` machine type, with **AMD SEV** confidential computing enabled. This is the team's **TEE variant**.

The delta between the two — same hardware, same image, same model, only the SEV flag differs — is the *runtime cost of confidential computing for DNN inference*. That delta is the headline number for the project's confidential-computing analysis.

> **Methodology correction:** the team's original protocol specified `n2-standard-4`. AMD SEV is only available on AMD EPYC machine series on GCP (`n2d-*`, `c2d-*`, `c3d-*`). The `n2-*` series is Intel and does not support SEV. I used `n2d-standard-4` for both VMs to keep the matched-pair comparison valid. Flagged for the team's final report.

---

## What I deployed

The Columbia GCP organization enforces the `compute.vmExternalIpAccess` org policy, so **none of these VMs have an external IP**. All VMs sit in the default VPC and communicate over internal IPs. SSH happens via IAP TCP forwarding.

| | `triton-cpu-standard` | `triton-cpu-confidential` | `harness-runner` |
|---|---|---|---|
| GCP project | `applied-ml-cloud` | `applied-ml-cloud` | `applied-ml-cloud` |
| Machine type | n2d-standard-4 (4 vCPU AMD EPYC, 16 GB RAM) | n2d-standard-4 (4 vCPU AMD EPYC, 16 GB RAM) | e2-standard-2 (2 vCPU, 8 GB RAM) |
| Confidential compute | (off) | **AMD SEV** | (off — N/A) |
| Image | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |
| Boot disk | 50 GB pd-balanced | 50 GB pd-balanced | 20 GB pd-balanced |
| Maintenance policy | MIGRATE (default) | TERMINATE (required for SEV) | MIGRATE (default) |
| Zone | us-central1-a | us-central1-a | us-central1-a |
| External IP | none (org policy) | none (org policy) | none (org policy) |
| Internal IP | _(set at run time)_ | _(set at run time)_ | _(set at run time)_ |
| Role | Team CPU baseline + serves Triton | TEE variant + serves Triton | Runs the harness against the two Triton VMs |

The two Triton VMs run Triton Inference Server in a Docker container with the model baked into a host volume. Both serve the byte-identical ONNX file (verified via SHA-256 against the team's committed hash in `model/resnet50.onnx.sha256`).

The harness runs **on the runner VM, not on my laptop**. This is required because the Triton VMs have no external IP, and tunneling the harness through IAP from my laptop would distort latency and concurrency measurements (IAP adds 5–15 ms per request and multiplexes connections through Google's edge proxy). Running the harness inside the same VPC gives clean internal-IP routing.

---

## What I ran

### One-time setup (from my laptop)

```bash
gcloud config configurations activate amlc          # yc4670@columbia.edu / applied-ml-cloud
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a
gcloud services enable compute.googleapis.com \
                       monitoring.googleapis.com \
                       logging.googleapis.com
bash confidential/firewall.sh   # IAP SSH (35.235.240.0/20:22) + internal Triton (10.0.0.0/8:8000)
```

### Provisioning all three VMs (laptop)

```bash
# Triton VMs (no external IP)
bash confidential/deploy_vm.sh     applied-ml-cloud standard
bash confidential/deploy_vm.sh     applied-ml-cloud confidential
# Harness-runner (no external IP, in same VPC)
bash confidential/deploy_runner.sh applied-ml-cloud

# Capture internal IPs
STD_IP=$(gcloud compute instances describe triton-cpu-standard \
  --zone us-central1-a --format='get(networkInterfaces[0].networkIP)')
CONF_IP=$(gcloud compute instances describe triton-cpu-confidential \
  --zone us-central1-a --format='get(networkInterfaces[0].networkIP)')
echo "Standard:     $STD_IP"
echo "Confidential: $CONF_IP"
```

Each Triton VM's GCE startup script takes ~5–7 minutes to install Docker, export the model, and start Triton. Watch progress without needing to SSH:

```bash
gcloud compute instances get-serial-port-output triton-cpu-standard \
  --zone us-central1-a | tail -50
```

### Connecting to a VM via IAP

Because no VM has an external IP, every SSH command needs `--tunnel-through-iap`:

```bash
gcloud compute ssh triton-cpu-standard --zone us-central1-a --tunnel-through-iap
# inside the VM:
docker ps                                           # confirm Triton running
sudo tail -f /var/log/triton-setup.log              # startup script log
docker logs triton --tail 100                       # Triton's stdout/stderr
```

### Uploading the harness to the runner VM

```bash
gcloud compute scp --tunnel-through-iap --zone us-central1-a --recurse \
  harness/ harness-runner:~
```

### Running the benchmark from the runner (inside the VPC)

```bash
gcloud compute ssh harness-runner --zone us-central1-a --tunnel-through-iap
# inside the runner:
curl -fsS http://${STD_IP}:8000/v2/health/ready  && echo "standard OK"
curl -fsS http://${CONF_IP}:8000/v2/health/ready && echo "confidential OK"

bash harness/run_all.sh http://${STD_IP}:8000  cpu          # → ~/results/cpu/
bash harness/run_all.sh http://${CONF_IP}:8000 confidential # → ~/results/confidential/
```

Each `run_all.sh` invocation runs the team protocol: 4 concurrency levels (1, 10, 50, 100) × 5 runs × 200 requests/client = 20 CSVs per VM, ~45–75 minutes total.

### Pulling the CSVs back to my laptop and committing

```bash
gcloud compute scp --tunnel-through-iap --zone us-central1-a --recurse \
  harness-runner:~/results ./
git add results/cpu/ results/confidential/ confidential/
git commit -m "Add confidential-computing slice: standard + SEV CPU benchmarks"
```

### Teardown (after results are committed)

```bash
bash confidential/teardown.sh applied-ml-cloud   # deletes all 3 VMs + firewall rules
```

---

## Expected results (predictions, to be replaced with measurements)

ResNet-50, FP32, batch 1, on n2d-standard-4 (4 vCPU AMD EPYC) via the ONNX Runtime CPU backend.

### Standard CPU VM — baseline prediction

| Concurrency | p50 latency (ms) | p95 latency (ms) | Throughput (req/s) |
|---:|---:|---:|---:|
| 1 | 80–150 | 120–200 | 7–12 |
| 10 | 100–180 | 200–350 | 25–40 |
| 50 | 800–1500 | 1500–2500 | 30–45 |
| 100 | 1800–3000 | 3000–5000 | 30–50 |

(Throughput plateaus around concurrency 50 because 4 vCPUs cap parallel inference; additional concurrency just queues.)

### Confidential VM — predicted delta

- **Latency overhead**: +5–15% across all percentiles, consistent with published AMD SEV overhead for memory-bound CPU workloads.
- **Throughput drop**: proportional to latency overhead at the saturation point (concurrency 50–100).
- **Cost premium**: ~6% list-price premium for SEV on n2d, compounded by the throughput drop → expect cost-per-1k-inferences to land **11–22% higher** than standard.

### Cost-per-1k-inferences prediction

| | Standard | Confidential | Delta |
|---|---:|---:|---:|
| n2d-standard-4 hourly | ~$0.135/hr | ~$0.143/hr | +6% |
| Throughput @ concurrency 100 | ~40 req/s | ~34–38 req/s | −5–15% |
| Cost per 1k inferences | ~$0.94 | ~$1.05–$1.15 | **+11–22%** |

---

## Actual results

> _Filled in after `harness/run_all.sh` completes for both VMs. Values are medians across the 5 runs at each concurrency level._

### Standard CPU VM (`results/cpu/`)

| Concurrency | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) |
|---:|---:|---:|---:|---:|
| 1 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 10 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 50 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 100 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### Confidential VM (`results/confidential/`)

| Concurrency | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) |
|---:|---:|---:|---:|---:|
| 1 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 10 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 50 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 100 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

### TEE overhead (the headline)

| Concurrency | Δ p50 (%) | Δ p95 (%) | Δ throughput (%) | Δ cost per 1k (%) |
|---:|---:|---:|---:|---:|
| 1 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 10 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 50 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 100 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

---

## What this tells us

For a workload as straightforward as ResNet-50 inference, the TEE overhead measured here is **the price of memory-encrypted execution under AMD SEV**. For inference workloads serving non-sensitive public data, this overhead is pure waste — the standard VM is strictly better on every axis. For workloads handling regulated data (medical imaging, financial document analysis, biometric matching), the same overhead is *the cost of compliance* — the question becomes whether that cost is acceptable, not whether it can be avoided.

The cleanly separated standard-vs-confidential measurement here is the team's main contribution to the confidential-computing dimension. Combined with Aashir's GPU numbers and the managed-service measurements from Shopnil and Ethan, it lets a reader pick a deployment that meets both their performance and their security requirements.

---

## How to reproduce

From `project2-benchmark/` with the active gcloud config set to `applied-ml-cloud`:

```bash
bash confidential/firewall.sh
bash confidential/deploy_vm.sh     applied-ml-cloud standard
bash confidential/deploy_vm.sh     applied-ml-cloud confidential
bash confidential/deploy_runner.sh applied-ml-cloud
# wait ~5–7 min for Triton startup scripts; ~2 min for the runner

gcloud compute scp --tunnel-through-iap --zone us-central1-a --recurse \
  harness/ harness-runner:~
gcloud compute ssh harness-runner --zone us-central1-a --tunnel-through-iap
# inside the runner:
bash harness/run_all.sh http://<std-internal-ip>:8000  cpu
bash harness/run_all.sh http://<conf-internal-ip>:8000 confidential
exit

gcloud compute scp --tunnel-through-iap --zone us-central1-a --recurse \
  harness-runner:~/results ./
bash confidential/teardown.sh applied-ml-cloud
```

Deeper rationale and risk notes live in [CLAUDE.md](CLAUDE.md). The shared experiment protocol that all four teammates run against is in [../README.md](../README.md) (top-level).
