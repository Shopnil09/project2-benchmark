"""
Cloud Functions 2nd gen entry point — ResNet-50 inference, Triton-HTTP-compatible.

Runtime: Python 3.11
Hardware: 4 vCPU / 8 GB (matches Shopnil's Cloud Run sizing for parity)
Entry point: triton_handler

This module speaks Triton's KServe v2 HTTP API on the wire so the shared
benchmarking harness (harness/harness.py) targets this endpoint without
modification. Internally it runs onnxruntime — Cloud Functions cannot run
Triton (no Docker).

Routes implemented (only what tritonclient[http] uses):
  GET  /v2/health/live           → 200
  GET  /v2/health/ready          → 200
  GET  /v2/models/<name>/ready   → 200 if name matches
  GET  /v2/models/<name>         → JSON model metadata
  POST /v2/models/<name>/infer   → inference (binary or JSON tensor I/O)
"""

import json
import logging
import time
from pathlib import Path

import functions_framework
import numpy as np
import onnxruntime as ort
from flask import Request, Response


MODEL_NAME    = "resnet50"
MODEL_VERSION = "1"
INPUT_NAME    = "input"
OUTPUT_NAME   = "output"
INPUT_SHAPE   = [1, 3, 224, 224]
INPUT_DTYPE   = "FP32"
OUTPUT_SHAPE  = [1, 1000]
OUTPUT_DTYPE  = "FP32"

MODEL_PATH = Path(__file__).parent / "resnet50.onnx"

# Triton datatype string → (numpy dtype, byte width)
_TRITON_DTYPE = {
    "FP32":  (np.float32, 4),
    "FP16":  (np.float16, 2),
    "INT64": (np.int64,   8),
    "INT32": (np.int32,   4),
    "UINT8": (np.uint8,   1),
}


# -----------------------------------------------------------------------------
# Cold-start init — runs once per instance, before the first request
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

_init_start = time.perf_counter()
log.info("Cold start: loading ONNX model from %s", MODEL_PATH)
_session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
COLD_START_MS = (time.perf_counter() - _init_start) * 1000.0
COLD_START_AT_UNIX = time.time()
log.info("Cold start complete in %.1f ms", COLD_START_MS)


# -----------------------------------------------------------------------------
# Triton v2 request/response codec
# -----------------------------------------------------------------------------
def _parse_request(body: bytes, header_len: int):
    """
    Parse a Triton v2 inference request. Handles two encodings:
      (a) Pure JSON: tensor data in inputs[i].data
      (b) Binary mixed (the tritonclient default): body[:header_len] is JSON,
          body[header_len:] is the concatenated binary blobs in input order,
          each input's size given by inputs[i].parameters.binary_data_size.

    Returns (dict[input_name -> np.ndarray], parsed_request_dict).
    """
    if header_len > 0:
        header_json = body[:header_len].decode("utf-8")
        binary_blob = body[header_len:]
    else:
        header_json = body.decode("utf-8")
        binary_blob = b""

    parsed = json.loads(header_json)
    tensors = {}
    offset = 0

    for spec in parsed.get("inputs", []):
        np_dtype, _ = _TRITON_DTYPE[spec["datatype"]]
        binary_size = spec.get("parameters", {}).get("binary_data_size")

        if binary_size is not None:
            chunk = binary_blob[offset:offset + binary_size]
            tensor = np.frombuffer(chunk, dtype=np_dtype).copy().reshape(spec["shape"])
            offset += binary_size
        else:
            tensor = np.array(spec["data"], dtype=np_dtype).reshape(spec["shape"])

        tensors[spec["name"]] = tensor

    return tensors, parsed


def _build_response(output: np.ndarray, request: dict) -> Response:
    """
    Build a Triton v2 inference response, mirroring the request's encoding.
    Defaults to binary output (what tritonclient requests by default).
    """
    out_specs   = request.get("outputs", [{"name": OUTPUT_NAME}])
    binary_data = out_specs[0].get("parameters", {}).get("binary_data", True)

    output_header = {
        "name":     OUTPUT_NAME,
        "datatype": OUTPUT_DTYPE,
        "shape":    list(output.shape),
    }
    body_json = {
        "model_name":    MODEL_NAME,
        "model_version": MODEL_VERSION,
        "outputs":       [output_header],
    }
    if "id" in request:
        body_json["id"] = request["id"]

    if binary_data:
        output_bytes = output.astype(np.float32).tobytes()
        output_header["parameters"] = {"binary_data_size": len(output_bytes)}
        header_bytes = json.dumps(body_json).encode("utf-8")
        return Response(
            response=header_bytes + output_bytes,
            status=200,
            content_type="application/octet-stream",
            headers={"Inference-Header-Content-Length": str(len(header_bytes))},
        )

    output_header["data"] = output.astype(np.float32).flatten().tolist()
    return Response(
        response=json.dumps(body_json),
        status=200,
        content_type="application/json",
    )


# -----------------------------------------------------------------------------
# Route metadata
# -----------------------------------------------------------------------------
_MODEL_METADATA = {
    "name":     MODEL_NAME,
    "versions": [MODEL_VERSION],
    "platform": "onnxruntime_onnx",
    "inputs":   [{"name": INPUT_NAME,  "datatype": INPUT_DTYPE,  "shape": INPUT_SHAPE}],
    "outputs":  [{"name": OUTPUT_NAME, "datatype": OUTPUT_DTYPE, "shape": OUTPUT_SHAPE}],
}


# -----------------------------------------------------------------------------
# Entry point — Cloud Functions 2nd gen routes everything here
# -----------------------------------------------------------------------------
@functions_framework.http
def triton_handler(request: Request) -> Response:
    path, method = request.path or "/", request.method

    if method == "GET":
        if path in ("/v2/health/live", "/v2/health/ready",
                    f"/v2/models/{MODEL_NAME}/ready"):
            return Response("", status=200)
        if path == f"/v2/models/{MODEL_NAME}":
            return Response(json.dumps(_MODEL_METADATA),
                            status=200, content_type="application/json")

    if method == "POST" and path == f"/v2/models/{MODEL_NAME}/infer":
        header_len = int(request.headers.get("Inference-Header-Content-Length", "0"))
        tensors, parsed = _parse_request(request.get_data(), header_len)
        output = _session.run([OUTPUT_NAME], {INPUT_NAME: tensors[INPUT_NAME]})[0]
        return _build_response(output, parsed)

    log.warning("Unhandled %s %s", method, path)
    return Response(json.dumps({"error": f"Not Found: {method} {path}"}),
                    status=404, content_type="application/json")
