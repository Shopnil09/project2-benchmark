"""
Export ResNet-50 to ONNX for the GCP inference benchmarking project.

Produces a SINGLE self-contained file:
  resnet50.onnx        — graph + weights, ~98 MB, FP32, shape [1, 3, 224, 224]
  resnet50.onnx.sha256 — integrity hash for verification

Usage:
  python model/export_model.py

Run from the repo root.
"""

import hashlib
import os
import sys

import numpy as np
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights


# ----- Configuration (matches PROTOCOL.md) ------------------------------------
OUTPUT_PATH = "resnet50.onnx"
HASH_PATH = "model/resnet50.onnx.sha256"
INPUT_SHAPE = (1, 3, 224, 224)
OPSET_VERSION = 18               # legacy exporter handles 18 cleanly; Triton 24.01 supports it
INPUT_NAME = "input"
OUTPUT_NAME = "output"


def cleanup_existing_artifacts():
    """Remove any prior exports so we start clean (avoids stale .onnx.data files)."""
    for path in [OUTPUT_PATH, OUTPUT_PATH + ".data", "resnet50.onnx_data"]:
        if os.path.exists(path):
            os.remove(path)
            print(f"  Removed stale: {path}")


def export_model():
    """Load pretrained ResNet-50 and export to ONNX as a SINGLE self-contained file."""
    print("Loading ResNet-50 with ImageNet pretrained weights...")
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.eval()  # freeze BatchNorm running stats, disable Dropout

    dummy_input = torch.randn(*INPUT_SHAPE, dtype=torch.float32)

    print(f"Exporting to {OUTPUT_PATH} (opset {OPSET_VERSION}, legacy exporter)...")
    # dynamo=False forces the legacy TorchScript-based exporter, which produces
    # a single self-contained .onnx file for models under 2 GB. The new dynamo
    # exporter (default in torch 2.5+) emits external .data files even for
    # small models, which we don't want.
    torch.onnx.export(
        model,
        dummy_input,
        OUTPUT_PATH,
        export_params=True,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamo=False,
    )

    # Verify single-file output
    data_file = OUTPUT_PATH + ".data"
    if os.path.exists(data_file):
        print(f"\nERROR: external data file {data_file} was created.")
        print("The legacy exporter still produced two files — unexpected for ResNet-50.")
        sys.exit(1)

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 ** 2)
    print(f"  Wrote {OUTPUT_PATH} ({size_mb:.1f} MB) — single file confirmed")
    return model, dummy_input


def verify_onnx_model(pytorch_model, sample_input):
    """Sanity-check the exported model by comparing PyTorch vs ONNX outputs."""
    try:
        import onnx
        import onnxruntime
    except ImportError:
        print("WARNING: onnx and/or onnxruntime not installed. Skipping verification.")
        print("         Install with: pip install onnx onnxruntime")
        return

    print("\nVerifying ONNX model...")

    # 1. Structural check
    onnx_model = onnx.load(OUTPUT_PATH)
    onnx.checker.check_model(onnx_model)
    print("  Structure: valid")

    # 2. Numerical check
    with torch.no_grad():
        torch_output = pytorch_model(sample_input).numpy()

    ort_session = onnxruntime.InferenceSession(OUTPUT_PATH, providers=["CPUExecutionProvider"])
    onnx_output = ort_session.run([OUTPUT_NAME], {INPUT_NAME: sample_input.numpy()})[0]

    max_diff = np.max(np.abs(torch_output - onnx_output))
    if max_diff < 1e-4:
        print(f"  Numerical match: max diff = {max_diff:.2e} (excellent)")
    elif max_diff < 1e-3:
        print(f"  Numerical match: max diff = {max_diff:.2e} (acceptable)")
    else:
        print(f"  WARNING: max diff = {max_diff:.2e} — outputs differ more than expected")
        sys.exit(1)

    assert onnx_output.shape == (1, 1000), f"Unexpected output shape: {onnx_output.shape}"
    print(f"  Output shape: {onnx_output.shape} (1000 ImageNet classes)")


def write_hash():
    """Write a SHA-256 of the .onnx file."""
    print("\nComputing SHA-256 hash...")
    h = hashlib.sha256()
    with open(OUTPUT_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    digest = h.hexdigest()

    os.makedirs(os.path.dirname(HASH_PATH), exist_ok=True)
    with open(HASH_PATH, "w") as f:
        f.write(f"{digest}  {OUTPUT_PATH}\n")
    print(f"  {digest}")
    print(f"  Written to {HASH_PATH}")


def main():
    print("Cleaning up any prior exports...")
    cleanup_existing_artifacts()

    pytorch_model, sample_input = export_model()
    verify_onnx_model(pytorch_model, sample_input)
    write_hash()

    print("\nDone. Next step:")
    print(f"  cp {OUTPUT_PATH} cloudrun/model_repository/resnet50/1/model.onnx")


if __name__ == "__main__":
    main()