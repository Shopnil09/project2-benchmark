"""
Smoke test for the Cloud Run Triton endpoint.
Sends one real inference request and prints the top-5 predicted classes.

Usage:
  python cloudrun/test_inference.py --endpoint <url>

Example:
  python cloudrun/test_inference.py \
    --endpoint https://triton-resnet50-xxxxx-uc.a.run.app
"""

import argparse
import time
import numpy as np

try:
    import tritonclient.http as httpclient
except ImportError:
    raise ImportError("Install with: pip install tritonclient[http]")

try:
    from PIL import Image
except ImportError:
    raise ImportError("Install with: pip install Pillow")


# ImageNet normalization constants — must match what export_model.py uses
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Top-5 ImageNet class labels (just a small subset for display purposes)
# Full list: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
IMAGENET_CLASSES = {
    0:   "tench",
    1:   "goldfish",
    2:   "great white shark",
    207: "golden retriever",
    208: "Labrador retriever",
    281: "tabby cat",
    282: "tiger cat",
    285: "Egyptian cat",
    291: "lion",
    292: "tiger",
    340: "zebra",
    386: "elephant",
    388: "giant panda",
}


def build_input_tensor():
    """
    Build a synthetic 224x224x3 input tensor.
    Uses random pixel values normalized with ImageNet mean/std.
    We use a synthetic image (not a real photo) because we're just
    verifying the model accepts the input and returns output — not
    checking prediction accuracy.
    """
    # Random RGB image, pixel values in [0, 1]
    image = np.random.rand(224, 224, 3).astype(np.float32)

    # Normalize with ImageNet mean and std (channel-wise)
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    # Transpose from HWC (224, 224, 3) to CHW (3, 224, 224) — PyTorch convention
    image = image.transpose(2, 0, 1)

    # Add batch dimension: (3, 224, 224) → (1, 3, 224, 224)
    image = np.expand_dims(image, axis=0)

    return image


def run_inference(endpoint, verbose=True):
    """Send one inference request and return latency + top class index."""

    # Strip https:// prefix if present — tritonclient adds its own scheme
    url = endpoint.replace("https://", "").replace("http://", "")

    if verbose:
        print(f"Connecting to: {url}")

    # ssl=True because Cloud Run endpoints are HTTPS
    client = httpclient.InferenceServerClient(url=url, ssl=True)

    # Check server is healthy before sending inference
    if not client.is_server_ready():
        raise RuntimeError("Server is not ready. Check the endpoint URL and deployment status.")
    if verbose:
        print("  Server: READY")

    # Check the model specifically is loaded
    if not client.is_model_ready("resnet50"):
        raise RuntimeError("Model 'resnet50' is not ready. Check config.pbtxt and model.onnx.")
    if verbose:
        print("  Model:  READY")

    # Build input tensor
    input_data = build_input_tensor()

    # Wrap in Triton's input format
    triton_input = httpclient.InferInput("input", input_data.shape, "FP32")
    triton_input.set_data_from_numpy(input_data)

    # Define expected output
    triton_output = httpclient.InferRequestedOutput("output")

    # Send request and time it
    if verbose:
        print("\nSending inference request...")
    start = time.perf_counter()
    response = client.infer(
        model_name="resnet50",
        inputs=[triton_input],
        outputs=[triton_output]
    )
    latency_ms = (time.perf_counter() - start) * 1000

    # Parse response
    output_data = response.as_numpy("output")  # shape: (1, 1000)

    if verbose:
        print(f"  Latency:       {latency_ms:.1f} ms")
        print(f"  Output shape:  {output_data.shape}")

        # Top-5 class indices by logit value
        top5_indices = np.argsort(output_data[0])[::-1][:5]
        print("\nTop-5 predictions (by logit score):")
        for rank, idx in enumerate(top5_indices):
            label = IMAGENET_CLASSES.get(idx, f"class_{idx}")
            score = output_data[0][idx]
            print(f"  {rank+1}. [{idx:4d}] {label:<30s}  score: {score:.4f}")

    return latency_ms, output_data


def main():
    parser = argparse.ArgumentParser(description="Triton inference smoke test")
    parser.add_argument(
        "--endpoint",
        required=True,
        help="Cloud Run endpoint URL, e.g. https://triton-resnet50-xxxxx-uc.a.run.app"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of requests to send (default: 3, to see latency variance)"
    )
    args = parser.parse_args()

    print(f"{'='*50}")
    print(f"Triton Inference Smoke Test")
    print(f"Endpoint: {args.endpoint}")
    print(f"{'='*50}\n")

    latencies = []
    for i in range(args.runs):
        if i > 0:
            print(f"\n--- Run {i+1} ---")
        latency, _ = run_inference(args.endpoint, verbose=(i == 0))
        latencies.append(latency)
        if i > 0:
            print(f"  Latency: {latency:.1f} ms")

    if args.runs > 1:
        print(f"\n{'='*50}")
        print(f"Summary over {args.runs} runs:")
        print(f"  First request (cold):  {latencies[0]:.1f} ms")
        print(f"  Min:                   {min(latencies):.1f} ms")
        print(f"  Max:                   {max(latencies):.1f} ms")
        print(f"  Avg:                   {sum(latencies)/len(latencies):.1f} ms")
        print(f"{'='*50}")

    print("\nSmoke test PASSED ✅")


if __name__ == "__main__":
    main()