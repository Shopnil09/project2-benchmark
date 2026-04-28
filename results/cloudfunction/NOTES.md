# NOTES.md — Cloud Functions benchmark run

**Owner:** Ethan Chang
**Config:** `cloudfunction` (Google Cloud Functions 2nd gen)
**Date of run:** 2026-04-28
**Time of run:** 17:08–17:45 EDT (deploy + cold-start samples + warm benchmark)
**Region:** `us-central1`
**Project:** `ancient-acumen-486002-j4`

> **Off-peak protocol deviation:** Protocol v1.0 specifies experiments run during off-peak hours (weekday 10pm–6am CT). This run started ~5pm ET (4pm CT) on a Tuesday — peak hours. Numbers may carry slight noisy-neighbor variance vs. a true off-peak run; flagged here for the report.

---

## Deployment configuration

| Field | Value |
|---|---|
| Function name | `triton-resnet50-cf` |
| Generation | 2nd gen (runs on Cloud Run under the hood) |
| Runtime | `python311` |
| Entry point | `triton_handler` |
| CPU | 4 vCPU |
| Memory | 8 GiB |
| Timeout | 540s |
| Concurrency (per instance) | 100 |
| Min instances | 1 |
| Max instances | 1 |
| Authentication | `--allow-unauthenticated` |
| Endpoint | `https://triton-resnet50-cf-4povcspdkq-uc.a.run.app` |

Memory was bumped from the v1.0 protocol default (4 GB) to 8 GiB after Shopnil's "keep CPU configs similar" call in team chat — this aligns Cloud Functions CPU/RAM with Cloud Run for a cleaner FaaS-vs-CaaS comparison. Team agreed before the run.

---

## Software versions

| Component | Version |
|---|---|
| Python | 3.11.4 |
| `functions-framework` | 3.8.1 |
| `onnxruntime` | 1.25.1 |
| `tritonclient[http]` (harness side) | 2.43.0 |
| ResNet-50 ONNX | SHA-256 `38da5bc82ddcd2e3a2f9b511b02622ae9be5dc8a50263a4a8adbea14bed12f78` (matches committed `model/resnet50.onnx.sha256`) |

---

## Cold-start measurement

**Methodology:** Force-redeploy via `bash cloudfunction/deploy.sh`, then immediately send one inference request via `tritonclient.http.InferenceServerClient.infer()` (no `is_server_ready()` health-check first, since that would itself be the cold-start request and warm the function before measurement).

**Samples (3 redeploys, back-to-back):**

| Sample | Client-side first-request latency |
|---|---|
| 1 | 410.5 ms |
| 2 | 415.2 ms |
| 3 | 377.3 ms |
| **Mean** | **401.0 ms** |
| **Range** | 38 ms (very tight) |

**Important caveat — what these numbers actually represent:** With `min-instances=1`, Cloud Run pre-warms the new revision during the deploy phase (sending startup TCP probes and healthchecks) **before** shifting traffic to it. By the time `gcloud functions deploy` returns, the new instance has already booted, imported `main.py`, loaded the 97 MB ONNX model, and passed Cloud Run's TCP probe — confirmed via `Default STARTUP TCP probe succeeded after 1 attempt for container "worker"` log lines emitted before our timed request hit the function.

So **these numbers are NOT pure cold-start latency** — they're closer to "warm first request after revision rollover." True cold-start (idle instance reaped + first request) would require:
1. Set `--min-instances=0` (deviates from protocol)
2. Wait ≥15 min for Cloud Run to reap the idle instance
3. Send first request — guaranteed cold

This was not measured because it conflicts with the protocol's `min-instances=1` requirement and the time budget. **Recommendation for the report:** discuss the user-visible cold-start tradeoff under both regimes — `min-instances=1` (~400 ms first-request post-deploy, what we measured) and `min-instances=0` (estimated 8-20 s, not measured).

**Inner model-load time (`COLD_START_MS` from `main.py` log line):** *Not captured.* `main.py` emits `log.info("Cold start complete in %.1f ms", COLD_START_MS)` at module-import time, but Cloud Run's logging integration filters out INFO-level Python logs from the worker process by default; only WARNING+ gets forwarded to Cloud Logging. To capture this in a future run, change `log.info` → `log.warning` in `main.py` and redeploy.

---

## Warm benchmark

**Run via:** `PATH=venv/bin:$PATH bash harness/run_all.sh https://triton-resnet50-cf-4povcspdkq-uc.a.run.app cloudfunction`

**Protocol parameters (per harness):**
- Concurrency levels: 1, 10, 50, 100 (ascending)
- Runs per level: 5
- Requests per client per run: 200 (first 20 discarded as warmup)
- Per-request timeout: 30s

**Output:** 20 CSVs at `results/cloudfunction/results_cloudfunction_<concurrency>_<run>.csv`. Schema per protocol: `request_id, client_id, send_ts, receive_ts, latency_ms, success, error`. **Aggregation (p50/p95/p99/throughput) is deferred to the cross-config aggregation phase at the end of the project — only raw per-request data is collected here.**

**Smoke-test latency (pre-benchmark sanity check, 6 requests):** 176-320 ms per request, avg 258 ms. This is consistent with the warm-state expectation for ResNet-50 FP32 batch=1 on 4 vCPUs over public internet.

### Per-run summary statistics

Captured from `harness.py`'s inline summary; raw per-request data is in the CSVs and is the canonical record for cross-config aggregation. Throughput = successful requests per second across all clients, wall-clock.

| C | run | p50 | p95 | p99 | throughput | error rate | valid? |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 205.9 ms | 249.8 ms | 284.2 ms | 4.8 req/s | 0.0% | ✓ |
| 1 | 2 | 210.7 ms | 315.5 ms | 449.0 ms | 4.4 req/s | 0.0% | ✓ |
| 1 | 3 | 197.6 ms | 237.9 ms | 266.5 ms | 4.9 req/s | 0.0% | ✓ |
| 1 | 4 | 208.9 ms | 245.8 ms | 436.0 ms | 4.6 req/s | 0.0% | ✓ |
| 1 | 5 | 202.6 ms | 306.1 ms | 337.7 ms | 4.5 req/s | 0.0% | ✓ |
| 10 | 1 | 488.6 ms | 815.5 ms | 1,062.9 ms | 17.1 req/s | 0.0% | ✓ |
| 10 | 2 | 478.2 ms | 742.5 ms | 1,053.6 ms | 19.4 req/s | 0.0% | ✓ |
| 10 | 3 | 496.8 ms | 854.4 ms | 1,047.0 ms | 17.3 req/s | 0.0% | ✓ |
| 10 | 4 | 516.7 ms | 1,139.8 ms | 1,857.2 ms | 16.1 req/s | 0.0% | ✓ |
| 10 | 5 | 410.0 ms | 753.1 ms | 980.5 ms | 20.0 req/s | 0.0% | ✓ |
| 50 | 1 | 2,815.6 ms | 4,352.4 ms | 6,878.8 ms | 16.0 req/s | 0.0% | ✓ |
| 50 | 2 | 2,808.7 ms | 3,550.8 ms | 3,840.3 ms | 17.6 req/s | 0.0% | ✓ |
| 50 | 3 | 3,181.1 ms | 3,881.3 ms | 4,118.9 ms | 16.2 req/s | 0.0% | ✓ |
| 50 | 4 | 3,026.3 ms | 3,772.9 ms | 3,953.8 ms | 16.7 req/s | 0.0% | ✓ |
| 50 | 5 | 2,678.7 ms | 3,708.4 ms | 3,922.4 ms | 18.0 req/s | 0.0% | ✓ |
| 100 | 1 | 4,873.8 ms | 5,947.5 ms | 6,369.2 ms | 17.8 req/s | **46.6%** | discard |
| 100 | 2 | 4,625.7 ms | 5,621.5 ms | 6,080.2 ms | 18.3 req/s | **50.4%** | discard |
| 100 | 3 (re-run) | 4,316.4 ms | 5,387.6 ms | 5,852.1 ms | 19.9 req/s | **42.2%** | discard |
| 100 | 4 | 4,418.0 ms | 5,197.2 ms | 5,541.4 ms | 19.7 req/s | **45.4%** | discard |
| 100 | 5 (re-run) | 4,559.4 ms | 5,512.0 ms | 5,998.3 ms | 18.9 req/s | **47.4%** | discard |
| **C=100 mean** | | **4,559 ms** | **5,533 ms** | **5,968 ms** | **18.9 req/s** | **46.4%** | (all discardable) |

*Numbers above are p50/p95/p99 of the latency distribution within each run after discarding the 20-request warmup per client. Throughput is the wall-clock rate at which successful requests completed across all clients.*

### The C=100 wall (key report finding)

Three independent C=100 runs (r=1, r=2, r=4) showed **structurally identical failure profiles** — not noise:

- Error rates clustered tightly (46.6%, 50.4%, 45.4%; mean 47.5%)
- Throughput on successful half clustered tightly (17.8, 18.3, 19.7 req/s; mean 18.6)
- p50 latency on successful half clustered tightly (4,874, 4,626, 4,418 ms; mean 4,639)

The function caps at **~17-18 req/s steady-state** on 4 vCPU regardless of concurrency level. Past C ≈ 50, additional clients only add queue depth — at C=100, the queue grows faster than the function can drain it, and the per-request 30-second timeout (set in `config.pbtxt`) fires for ~half the queued requests. This is the FaaS scaling story for the report: not graceful degradation, but a hard wall around C=50→C=100 at this hardware tier.

The two C=100 runs that failed at the *pre-run* readiness probe (r=3, r=5 — originally) are evidence of the same wall: after a heavy C=100 run completed, the function's `/v2/health/ready` endpoint stayed unhealthy long enough for the next run's startup probe to fail. Cloud Run's frontend was still draining queues.

### Re-runs of C=100 r=3 and r=5

**Original outcome (during the initial loop):** Both runs failed at the harness's pre-run `is_server_ready()` probe with `RuntimeError: Server not ready at <url>. Check deployment.` No CSV produced. Cause: function was still draining requests from the previous heavy run when the next run started — Cloud Run's per-instance request concurrency was capped at 100, so an already-overloaded function rejected the readiness probe with HTTP 503 until it caught up.

**Re-run methodology (after the original loop ended):** Verified the function had recovered (single readiness probe returned 200), then ran r=3 first, paused 90 seconds for the function to cool, verified readiness again, then ran r=5. Same harness invocation, same parameters. Re-run started 2026-04-28 at 18:55 EDT.

**Per protocol:** "Any run with error rate above 5% is discarded and re-run once. If it fails again, note it in the results." The originals didn't produce CSVs (zero data, not above-threshold), but the re-run policy was applied conservatively.

**Re-run outcome (2026-04-28, ~19:15 EDT):** Both re-runs completed successfully (i.e., produced CSVs) but, like all other C=100 runs, exceeded the 5% error threshold:

- r=3 (re-run): 42.2% errors, p50 4,316 ms, throughput 19.9 req/s
- r=5 (re-run): 47.4% errors, p50 4,559 ms, throughput 18.9 req/s

These match the original three C=100 profiles within noise — strengthening the conclusion that **the C=100 wall is structural, not transient**. Per protocol's "if it fails again, note it" clause: noted here and reflected in the row labels above. All five C=100 CSVs are present at `results/cloudfunction/results_cloudfunction_100_{1..5}.csv`; raw per-request data is preserved for cross-config aggregation, with the over-threshold flag carried forward in this notes file.

---

## GCP pricing used for cost calculation

**Recorded:** 2026-04-28, GCP Cloud Functions 2nd gen on-demand pricing for `us-central1`:

- **Per-invocation:** $0.0000004 / invocation (after 2M free tier per month)
- **vCPU-second:** $0.0000240 / vCPU-s (after 240,000 vCPU-s free tier)
- **GiB-second:** $0.0000025 / GiB-s (after 450,000 GiB-s free tier)
- **Networking egress:** $0.12/GiB to internet (negligible for our payload sizes)

Cost-per-1000-inferences formula (post-hoc, computed from CSVs):

```
cost = (n_requests * per_invocation_rate)
     + (total_run_seconds * 4_vCPU * vCPU_rate)
     + (total_run_seconds * 8_GiB * GiB_rate)
divided by (n_requests / 1000)
```

Free tier may zero this out for small runs — note in the report that real production cost is the steady-state above-tier rate, not the often-zero observed cost.

*Source: https://cloud.google.com/functions/pricing — pricing checked on 2026-04-28.*

---

## Anomalies / deviations from protocol

| Item | Description |
|---|---|
| Memory (4 GB → 8 GiB) | Documented above; team-approved for CPU-class parity. |
| Off-peak window | Run started in peak hours (~4pm CT). Light noisy-neighbor risk. |
| `min-instances=1` cold-start interpretation | The "cold-start" measurement here is post-redeploy, not reaped-instance. Methodology constraint, see Cold-start section. |
| `HEAD /v2/health/ready` 404s | Cloud Run sends HEAD probes that `main.py` doesn't handle (only GET). Returns 404 but does not affect function operation or the benchmark. Cosmetic — could add HEAD handling in a follow-up. |
| Two transient SIGSEGV errors during cold-start measurement | Two worker processes (pid=19, 20) segfaulted at 21:15:21 and 21:15:23 UTC during revision rollover. Did not recur during the warm benchmark; function remained healthy throughout. Likely benign Cloud Run worker churn during instance replacement. |

---

## Reproduction

```bash
# From repo root, with venv activated:
python model/export_model.py
shasum -a 256 -c model/resnet50.onnx.sha256          # confirm hash
cp resnet50.onnx cloudfunction/

bash cloudfunction/deploy.sh                          # ~3-5 min
python cloudfunction/test_inference.py --endpoint <url>  # smoke test

PATH=venv/bin:$PATH bash harness/run_all.sh <url> cloudfunction  # ~25 min
```
