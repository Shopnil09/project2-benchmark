# CLAUDE.md — Serverless Deployment (Cloud Functions 2nd gen)

**Owner:** Ethan Chang
**Project:** DNN Inference Benchmarking on GCP — Serverless slice
**Model:** ResNet-50 (ONNX, FP32, batch size 1)
**Serving framework:** `onnxruntime` behind a Triton-HTTP-compatible adapter
**Region:** us-central1

---

## GCP environment

Inherits the project-level setup verified by Shopnil (see [`cloudrun/CLAUDE.md`](../cloudrun/CLAUDE.md)):

- **Project:** `ancient-acumen-486002-j4`
- **APIs required for this slice (already enabled):** `cloudfunctions.googleapis.com`, `run.googleapis.com`, `cloudbuild.googleapis.com`, `artifactregistry.googleapis.com`, `logging.googleapis.com`, `monitoring.googleapis.com`. Cloud Functions 2nd gen runs on Cloud Run under the hood, so the run/build APIs already enabled for the container slice cover this one.
- **Networking:** No special setup. Cloud Functions exposes a Google-managed HTTPS endpoint; no VPC, NAT, or VPC connector needed.
- **Authentication:** `--allow-unauthenticated` matches Shopnil's Cloud Run setup so the harness can hit the URL without auth tokens.

---

## Scope of this slice

One responsibility: **deploy ResNet-50 inference as a Cloud Function and measure its end-to-end performance, with cold-start as the headline metric.**

Does not own the harness (Shopnil's), the model export (canonical artifact in [`model/`](../model/)), or any other config. Cross-config aggregation is deferred to the end of the project.

---

## What Cloud Functions is (and isn't)

Cloud Functions 2nd gen is GCP's HTTP-triggered FaaS layer. You hand it a Python source directory + `requirements.txt`, it builds and runs your code on a managed runtime, scales it, and bills per request × duration × memory.

Under the hood, **2nd gen runs your code as a container on Cloud Run.** The practical difference vs. Shopnil's Cloud Run isn't the runtime — it's the developer surface: no Dockerfile, no model_repository layout, no port management. Source dir in, HTTPS URL out.

**Hardware:** CPU-only. No GPU. ResNet-50 inference runs on the 4 vCPUs allocated to the function (matched to Shopnil's Cloud Run sizing for parity).

**What this config measures:** the FaaS premium — the cost (in latency, in cold-start, in dollars per 1000 inferences) of trading away container ownership for a managed-source-code surface.

---

## Why no Triton

Cloud Functions 2nd gen accepts a Python source directory; you can't supply a custom container image (that's what Cloud Run is for). Triton Inference Server is distributed only as a Docker image and requires its own runtime, model repository layout, and process — none of which fit.

**The workaround:** [`main.py`](main.py) loads the ONNX model directly with `onnxruntime` and exposes a tiny HTTP adapter that **speaks Triton's KServe v2 protocol on the wire.** From the harness's perspective (which uses `tritonclient[http]`), this endpoint is indistinguishable from a real Triton server. From the function's perspective, it's a single Python file calling `session.run()`. This is what makes the four-way comparison fair — same wire protocol, same harness, only the runtime underneath differs.

The adapter handles the Triton binary tensor format (JSON header + raw tensor blob, with `Inference-Header-Content-Length`) because that's what `tritonclient` sends by default. JSON-only is supported as a fallback.

---

## The four things that will bite us if ignored

1. **Cold start dominates the first request.** Cloud Functions 2nd gen with `min-instances=1` keeps one warm copy alive, but the very first request after deploy (or after a long enough idle period) pays the full model-load cost — `ort.InferenceSession` on a 97 MB ONNX is the floor. **Measurement strategy below.**

2. **`.gcloudignore` overrides `.gitignore` — but only if it exists.** The repo `.gitignore` excludes `*.onnx`. Without an explicit `cloudfunction/.gcloudignore`, gcloud falls back to `.gitignore`, the deploy archive ships without `resnet50.onnx`, and the function fails at module import with `FileNotFoundError`. The committed [`cloudfunction/.gcloudignore`](.gcloudignore) inverts this — it's the file that *guarantees* the model gets uploaded.

3. **Source archive size matters.** The 97 MB ONNX is well under the 2nd gen 500 MB compressed limit, but the deploy command uploads the whole `cloudfunction/` directory to Cloud Storage on every deploy. Don't add scratch files there. The committed `.gcloudignore` excludes `deploy.sh`, `test_inference.py`, and `CLAUDE.md` from the upload — they're local-only.

4. **`min-instances=1 max-instances=1` does NOT eliminate cold start.** It eliminates **scale-up cold starts** during the benchmark. The very first deploy or any redeploy still costs one cold start. Cold-start measurement therefore requires forcing the boundary explicitly (force-redeploy or wait for the warm instance to be reaped, then time the first request).

---

## Required deployment flags

| Flag | Value | Why |
|---|---|---|
| `--gen2` | (flag) | 2nd gen runs on Cloud Run; gives us higher resource ceilings, longer timeout, and `min-instances` |
| `--runtime` | `python311` | Matches the local venv that exported and verified the ONNX |
| `--entry-point` | `triton_handler` | The `@functions_framework.http`-decorated function in `main.py` |
| `--source` | `cloudfunction/` | Uploads `main.py`, `requirements.txt`, `resnet50.onnx` |
| `--memory` | `8Gi` | Per the team-aligned CPU-class sizing (4vCPU/8GB) — bumped from the v1.0 protocol's 4GB after Shopnil's "keep configs similar" call |
| `--cpu` | `4` | Same |
| `--timeout` | `540s` | 2nd gen HTTP function max; wide margin over the 30s per-request protocol timeout |
| `--min-instances` | `1` | Keep one instance warm; eliminate scale-up cold starts during the benchmark |
| `--max-instances` | `1` | Disable autoscaling; per protocol |
| `--concurrency` | `100` | Per-instance concurrency must be ≥ harness max concurrency or requests queue at the frontend before reaching the function |
| `--region` | `us-central1` | Per protocol |
| `--allow-unauthenticated` | (flag) | Matches Shopnil's Cloud Run; lets the harness hit the URL without tokens |

The deploy command is wrapped in [`deploy.sh`](deploy.sh).

---

## Cold-start measurement methodology

Cold-start is **the** metric for this slice. Measurement:

1. **Force a known cold start.** Two options:
   - Redeploy via `bash cloudfunction/deploy.sh` — guaranteed cold.
   - Update the function with a no-op env var change — `gcloud functions deploy ... --update-env-vars CHURN=$(date +%s)` — also cold without a full source rebuild.
2. **Time the first request immediately after the deploy completes.** [`test_inference.py`](test_inference.py) reports first-request latency separately for exactly this reason.
3. **Cross-reference with Cloud Logging** — `main.py` logs `Cold start complete in X ms` once per instance startup. The log line gives the model-load portion only; the first-request latency from the client side gives the end-to-end cold-start cost (model load + cold network + first inference).
4. **Repeat once per concurrency level** before the warm benchmark runs. Record one cold-start value per concurrency level (4 numbers total) in [`results/cloudfunction/NOTES.md`](../results/cloudfunction/NOTES.md). Cold-start is **not** part of the warm-benchmark CSV — it's a separate per-config one-time measurement.

---

## Performance expectations

ResNet-50, FP32, batch size 1, on 4 vCPUs via `onnxruntime`:

- **Cold start:** estimated 8-20 seconds end-to-end (model load + container init + first inference). This is the headline tradeoff — orders of magnitude worse than any other config.
- **Warm latency:** 80-200 ms per inference, similar ballpark to Shopnil's Cloud Run since the underlying hardware class is the same. Any meaningful gap vs. Cloud Run is the FaaS overhead (frontend routing, framework wrapper).
- **Throughput:** capped by single-instance + 100-concurrency setting; expect saturation at high concurrency levels.

These are estimates — the actual numbers come from the benchmark.

---

## What this config compares against

| Comparison | What the delta tells us |
|---|---|
| Cloud Functions vs. Shopnil's Cloud Run | The pure FaaS premium — same hardware class, same container runtime, only the source-vs-image surface differs |
| Cloud Functions vs. Eric's standard CPU VM | FaaS managed-runtime + cold-start cost vs. always-on VM |
| Cloud Functions vs. Aashir's GPU VM | Convenience-and-idle-cost tradeoff against raw inference speed |

Cold-start is the dimension where Cloud Functions loses badly to every other config. Warm-state cost-per-1000-inferences is the dimension where it wins (or should win — to be measured).

---

## Open items / decisions

- [x] ~~Memory tier — 4GB (v1.0 protocol) vs. 8GB~~ → **decided: 8GB to match Shopnil's Cloud Run for fair CPU-class comparison.** Team agreed in chat.
- [x] ~~Triton vs. raw HTTP server~~ → **decided: tiny in-process Triton-HTTP adapter** so the harness works unmodified. Code in [`main.py`](main.py), validated locally against `tritonclient[http]`.
- [x] ~~Model distribution — bundle in source vs. download from GCS~~ → **decided: bundle in source.** Same rationale as the canonical model artifact: deterministic from `model/export_model.py` + committed SHA, no need for a redundant cloud bucket. Adds ~97 MB to deploy archive — within 2nd gen limits.
- [ ] Authenticated invocation? Currently `--allow-unauthenticated` to match Cloud Run. Likely no for the duration of the study.
- [ ] Whether to set environment variables for ONNX runtime threading (`ORT_NUM_THREADS`). Default is "use all available CPUs" which should already saturate the 4 vCPU allocation — leave default unless the warm benchmark shows under-utilization in Cloud Logging metrics.

---

## Canonical model artifact

Same as the rest of the project — see top-level [`README.md`](../README.md) and [`model/`](../model/). Run `python model/export_model.py` from the repo root to produce `resnet50.onnx`, then `cp resnet50.onnx cloudfunction/` before deploying. Verify integrity with `shasum -a 256 -c model/resnet50.onnx.sha256`.

---

## Build, deploy, smoke test

```bash
# 1. Export the model and stage it (one-time, unless already done)
python model/export_model.py
cp resnet50.onnx cloudfunction/

# 2. Deploy
bash cloudfunction/deploy.sh

# 3. Smoke test (the deploy.sh output prints the URL — substitute it here)
python cloudfunction/test_inference.py --endpoint <function-url>

# 4. Measure cold start (force a redeploy, then immediately time a first request)
bash cloudfunction/deploy.sh
python cloudfunction/test_inference.py --endpoint <function-url> --runs 1

# 5. Warm benchmark (run after the function has handled at least one warmup request)
bash harness/run_all.sh <function-url> cloudfunction
```

The benchmark produces 20 CSVs in [`results/cloudfunction/`](../results/cloudfunction/) (4 concurrency levels × 5 runs). Cold-start numbers go in `NOTES.md` separately.
