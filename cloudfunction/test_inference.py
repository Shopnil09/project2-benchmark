"""
Smoke test for the deployed Cloud Function endpoint.

Sends N inference requests via tritonclient[http] — the SAME path the
shared harness uses — so a passing run here means the harness will work
against this endpoint without modification.

Usage:
  python cloudfunction/test_inference.py --endpoint <url>

Example:
  python cloudfunction/test_inference.py \\
    --endpoint https://triton-resnet50-cf-xxxxx-uc.a.run.app \\
    --runs 6

The first request is reported separately because cold-start latency
on Cloud Functions is the headline tradeoff for the serverless slice
of this study.
"""

import argparse
import ssl
import time

import numpy as np

try:
    import tritonclient.http as httpclient
except ImportError:
    raise ImportError("Install with: pip install tritonclient[http]")


# Must match the harness and main.py exactly
MODEL_NAME  = "resnet50"
INPUT_NAME  = "input"
OUTPUT_NAME = "output"
INPUT_SHAPE = [1, 3, 224, 224]
INPUT_DTYPE = "FP32"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_input_tensor() -> np.ndarray:
    """Build a deterministic FP32 input tensor matching the harness's payload."""
    rng = np.random.default_rng(seed=42)
    image = rng.random((224, 224, 3), dtype=np.float32)
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = image.transpose(2, 0, 1)
    return np.expand_dims(image, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Cloud Function inference smoke test")
    parser.add_argument(
        "--endpoint", required=True,
        help="Cloud Function HTTPS URL — output of `gcloud functions describe`",
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of requests to send (default: 5; 1st is reported separately)",
    )
    args = parser.parse_args()

    use_ssl = args.endpoint.startswith("https://")
    url = args.endpoint.replace("https://", "").replace("http://", "")

    print(f"{'='*55}")
    print(f"Cloud Function smoke test")
    print(f"Endpoint: {args.endpoint}")
    print(f"{'='*55}\n")

    # ssl_context_factory uses the modern SSLContext API, which auto-loads
    # the platform's default verify paths. Without it, geventhttpclient
    # falls back to the legacy ssl.wrap_socket() path with an empty trust
    # store, which fails verification even when the certs are installed.
    client = httpclient.InferenceServerClient(
        url=url,
        ssl=use_ssl,
        ssl_context_factory=ssl.create_default_context if use_ssl else None,
    )

    if not client.is_server_ready():
        raise RuntimeError("Server not ready. Check the URL and deployment.")
    print("  Server: READY")

    if not client.is_model_ready(MODEL_NAME):
        raise RuntimeError(f"Model '{MODEL_NAME}' not ready. Check main.py + resnet50.onnx.")
    print("  Model:  READY\n")

    input_data    = build_input_tensor()
    triton_input  = httpclient.InferInput(INPUT_NAME, INPUT_SHAPE, INPUT_DTYPE)
    triton_input.set_data_from_numpy(input_data)
    triton_output = httpclient.InferRequestedOutput(OUTPUT_NAME)

    latencies = []
    top5_first = None

    for i in range(args.runs):
        t0 = time.perf_counter()
        response = client.infer(
            model_name=MODEL_NAME,
            inputs=[triton_input],
            outputs=[triton_output],
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        output = response.as_numpy(OUTPUT_NAME)
        if output.shape != (1, 1000):
            raise RuntimeError(f"Unexpected output shape: {output.shape}")

        if i == 0:
            top5_first = np.argsort(output[0])[-5:][::-1].tolist()

        latencies.append(latency_ms)
        tag = "1st (may be cold)" if i == 0 else f"req {i+1}"
        print(f"  {tag:<20s}  {latency_ms:7.1f} ms")

    warm = latencies[1:]
    print(f"\n{'='*55}")
    print(f"First request:           {latencies[0]:7.1f} ms")
    if warm:
        print(f"Warm requests ({len(warm)}):       "
              f"min {min(warm):.1f}  avg {sum(warm)/len(warm):.1f}  max {max(warm):.1f} ms")
    print(f"Output top-5 indices:    {top5_first}")
    print(f"{'='*55}")
    print("\nSmoke test PASSED")


if __name__ == "__main__":
    main()
