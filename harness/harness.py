"""
Benchmarking harness for DNN Inference on GCP — Project 2
Shared tool used by all four teammates against their own deployments.

Usage:
  python harness/harness.py \
    --endpoint https://triton-resnet50-xxxxx-uc.a.run.app \
    --config-name cloudrun \
    --concurrency 1 \
    --run 1 \
    --requests 200 \
    --warmup 20 \
    --output results/

Output:
  results/results_cloudrun_1_1.csv

CSV schema (one row per request):
  request_id, client_id, send_ts, receive_ts, latency_ms, success, error

Authors: Shopnil Shahriar (owner), shared across team
"""

import argparse
import csv
import os
import platform
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    import tritonclient.http as httpclient
except ImportError:
    raise ImportError("Run: pip install tritonclient[http]")


# ---------------------------------------------------------------------------
# Constants — must match export_model.py and config.pbtxt exactly
# ---------------------------------------------------------------------------
MODEL_NAME   = "resnet50"
INPUT_NAME   = "input"
OUTPUT_NAME  = "output"
INPUT_SHAPE  = [1, 3, 224, 224]
INPUT_DTYPE  = "FP32"

# ImageNet normalization — same values as export_model.py
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class RequestRecord:
    """One row in the output CSV."""
    request_id: int
    client_id:  int
    send_ts:    float   # Unix timestamp (seconds), high-resolution
    receive_ts: float   # Unix timestamp (seconds), high-resolution
    latency_ms: float   # end-to-end client-side latency
    success:    bool    # False if any exception or HTTP error occurred
    error:      str     # empty string on success; exception message on failure


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------
def build_input_tensor() -> np.ndarray:
    """
    Build a synthetic FP32 input tensor matching the model's expected shape.
    Shape: [1, 3, 224, 224] — batch=1, RGB, 224x224
    Normalized with ImageNet mean and std, same as training preprocessing.
    We use the same random seed per process so all clients send
    identical payloads — payload content doesn't affect latency measurement.
    """
    rng = np.random.default_rng(seed=42)
    image = rng.random((224, 224, 3), dtype=np.float32)  # HWC, values in [0,1]
    image = (image - IMAGENET_MEAN) / IMAGENET_STD        # normalize
    image = image.transpose(2, 0, 1)                      # HWC → CHW
    image = np.expand_dims(image, axis=0)                 # → [1, 3, 224, 224]
    return image


# ---------------------------------------------------------------------------
# Single client worker
# ---------------------------------------------------------------------------
def run_client(
    client_id:    int,
    endpoint:     str,
    n_requests:   int,
    n_warmup:     int,
    use_ssl:      bool,
) -> List[RequestRecord]:
    """
    One concurrent client. Sends n_requests sequential HTTP inference
    requests. Discards the first n_warmup requests. Returns one
    RequestRecord per measured request.
    """
    # Strip scheme — tritonclient adds it based on ssl flag
    url = endpoint.replace("https://", "").replace("http://", "")

    # Each thread gets its own client connection
    client = httpclient.InferenceServerClient(url=url, ssl=use_ssl)

    # Pre-build input tensor and wrap it once (reuse across requests)
    input_data = build_input_tensor()
    triton_input = httpclient.InferInput(INPUT_NAME, INPUT_SHAPE, INPUT_DTYPE)
    triton_input.set_data_from_numpy(input_data)
    triton_output = httpclient.InferRequestedOutput(OUTPUT_NAME)

    records = []
    global_request_id = client_id * n_requests  # unique IDs across clients

    for i in range(n_requests):
        request_id = global_request_id + i
        send_ts    = time.perf_counter()
        send_wall  = time.time()

        try:
            response = client.infer(
                model_name=MODEL_NAME,
                inputs=[triton_input],
                outputs=[triton_output],
            )
            receive_ts   = time.perf_counter()
            receive_wall = time.time()
            latency_ms   = (receive_ts - send_ts) * 1000.0
            success      = True
            error        = ""

            # Validate output shape as a basic sanity check
            output = response.as_numpy(OUTPUT_NAME)
            if output.shape != (1, 1000):
                success = False
                error   = f"unexpected output shape: {output.shape}"

        except Exception as e:
            receive_wall = time.time()
            latency_ms   = (time.time() - send_wall) * 1000.0
            success      = False
            error        = str(e)

        # Skip warmup requests — don't record them
        if i < n_warmup:
            continue

        records.append(RequestRecord(
            request_id = request_id,
            client_id  = client_id,
            send_ts    = send_wall,
            receive_ts = receive_wall,
            latency_ms = latency_ms,
            success    = success,
            error      = error,
        ))

    return records


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------
def write_csv(records: List[RequestRecord], output_path: str) -> None:
    """Write all request records to a CSV file."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "request_id", "client_id",
            "send_ts", "receive_ts",
            "latency_ms", "success", "error"
        ])
        for r in records:
            writer.writerow([
                r.request_id,
                r.client_id,
                f"{r.send_ts:.6f}",
                f"{r.receive_ts:.6f}",
                f"{r.latency_ms:.3f}",
                int(r.success),
                r.error,
            ])


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def print_summary(records: List[RequestRecord], concurrency: int) -> None:
    """Print a quick summary after the run — not saved, just for visibility."""
    if not records:
        print("  No records to summarize.")
        return

    latencies   = [r.latency_ms for r in records if r.success]
    n_total     = len(records)
    n_success   = sum(1 for r in records if r.success)
    n_failed    = n_total - n_success
    error_rate  = (n_failed / n_total) * 100 if n_total > 0 else 0

    if latencies:
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        avg = float(np.mean(latencies))
    else:
        p50 = p95 = p99 = avg = 0.0

    # Throughput: total successful requests / total elapsed wall time
    if records:
        wall_start = min(r.send_ts    for r in records)
        wall_end   = max(r.receive_ts for r in records)
        elapsed    = wall_end - wall_start
        throughput = n_success / elapsed if elapsed > 0 else 0
    else:
        throughput = 0

    print(f"\n  Requests:     {n_total} total, {n_success} succeeded, {n_failed} failed")
    print(f"  Error rate:   {error_rate:.1f}%")
    print(f"  Latency p50:  {p50:.1f} ms")
    print(f"  Latency p95:  {p95:.1f} ms")
    print(f"  Latency p99:  {p99:.1f} ms")
    print(f"  Avg latency:  {avg:.1f} ms")
    print(f"  Throughput:   {throughput:.1f} req/s ({concurrency} concurrent clients)")

    if error_rate > 5.0:
        print(f"\n  ⚠️  ERROR RATE {error_rate:.1f}% EXCEEDS 5% THRESHOLD.")
        print(f"  Per protocol: discard this run and re-run once.")


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(
    endpoint:    str,
    config_name: str,
    concurrency: int,
    run_number:  int,
    n_requests:  int,
    n_warmup:    int,
    output_dir:  str,
) -> str:
    """
    Run the benchmark for one (config, concurrency, run) combination.
    Spawns `concurrency` threads, each sending `n_requests` sequential
    requests. Returns the path to the output CSV.
    """
    use_ssl = endpoint.startswith("https://")

    print(f"\n{'='*55}")
    print(f"Config:      {config_name}")
    print(f"Endpoint:    {endpoint}")
    print(f"Concurrency: {concurrency} clients")
    print(f"Requests:    {n_requests} per client ({n_warmup} warmup discarded)")
    print(f"Run:         {run_number}")
    print(f"{'='*55}")

    # Verify server is reachable before spawning threads
    url = endpoint.replace("https://", "").replace("http://", "")
    probe = httpclient.InferenceServerClient(url=url, ssl=use_ssl)
    if not probe.is_server_ready():
        raise RuntimeError(f"Server not ready at {endpoint}. Check deployment.")
    if not probe.is_model_ready(MODEL_NAME):
        raise RuntimeError(f"Model '{MODEL_NAME}' not ready. Check config.pbtxt.")
    print(f"\nServer and model: READY")
    print(f"Starting {concurrency} concurrent client(s)...\n")

    # Run all clients concurrently
    all_records: List[RequestRecord] = []
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(run_client, cid, endpoint, n_requests, n_warmup, use_ssl): cid
            for cid in range(concurrency)
        }
        for future in as_completed(futures):
            cid = futures[future]
            try:
                client_records = future.result()
                with lock:
                    all_records.extend(client_records)
                print(f"  Client {cid:3d} done — {len(client_records)} records")
            except Exception as e:
                print(f"  Client {cid:3d} FAILED: {e}")

    # Sort by send timestamp for clean CSV ordering
    all_records.sort(key=lambda r: r.send_ts)

    # Write CSV
    csv_filename = f"results_{config_name}_{concurrency}_{run_number}.csv"
    csv_path     = os.path.join(output_dir, csv_filename)
    write_csv(all_records, csv_path)
    print(f"\nWrote {len(all_records)} records → {csv_path}")

    # Print quick summary
    print_summary(all_records, concurrency)

    return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="GCP Inference Benchmarking Harness — Project 2"
    )
    parser.add_argument(
        "--endpoint", required=True,
        help="Triton server endpoint. Cloud Run: https://... | VM: http://IP:8000"
    )
    parser.add_argument(
        "--config-name", required=True,
        choices=["gpu", "cpu", "confidential", "cloudrun", "cloudfunction"],
        help="Deployment config name — used in output CSV filename"
    )
    parser.add_argument(
        "--concurrency", type=int, required=True,
        choices=[1, 10, 50, 100],
        help="Number of simultaneous clients"
    )
    parser.add_argument(
        "--run", type=int, required=True,
        help="Run number (1–5 per protocol)"
    )
    parser.add_argument(
        "--requests", type=int, default=200,
        help="Total requests per client including warmup (default: 200)"
    )
    parser.add_argument(
        "--warmup", type=int, default=20,
        help="Warmup requests to discard per client (default: 20)"
    )
    parser.add_argument(
        "--output", type=str, default="results/",
        help="Output directory for CSV files (default: results/)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = run_benchmark(
        endpoint    = args.endpoint,
        config_name = args.config_name,
        concurrency = args.concurrency,
        run_number  = args.run,
        n_requests  = args.requests,
        n_warmup    = args.warmup,
        output_dir  = args.output,
    )

    print(f"\nDone. CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()