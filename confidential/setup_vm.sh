#!/bin/bash
# confidential/setup_vm.sh
# Runs ON the VM as the GCE startup script. Installs Docker, exports
# the canonical ResNet-50 ONNX model, verifies the SHA-256, and starts
# Triton Inference Server in a container listening on TCP 8000.
#
# This script is identical for the standard and confidential VMs —
# the only difference is the host kernel/firmware (AMD SEV on/off).
# Inference behavior must be byte-identical; only timing differs.

set -euo pipefail
exec > >(tee -a /var/log/triton-setup.log) 2>&1

echo "==> [$(date -Is)] Starting Triton setup"

# Idempotency: if the model is already loaded and Triton is healthy, exit.
if curl -fsS http://localhost:8000/v2/health/ready >/dev/null 2>&1; then
  echo "==> Triton already healthy — skipping setup"
  exit 0
fi

# ---------- 1. System packages ---------------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates curl gnupg python3-pip python3-venv docker.io

systemctl enable --now docker

# ---------- 2. Python deps for the model export ----------------------
# Ubuntu 22.04 ships pip 22 which doesn't know --break-system-packages
# and isn't PEP 668-marked, so a plain `pip install` works. Upgrade pip
# first so we don't trip over old resolvers.
python3 -m pip install --no-cache-dir --upgrade pip
# torch>=2.5 is required: torch.onnx.export() gained the `dynamo` kwarg
# in 2.5; the team's export_model.py uses dynamo=False to force the
# single-file legacy exporter. CPU-only wheels keep the install small.
python3 -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==2.5.1" "torchvision==0.20.1"
python3 -m pip install --no-cache-dir "onnx" "onnxruntime" "numpy"

# ---------- 3. Pull the pinned Triton image (cache it eagerly) -------
TRITON_IMAGE="nvcr.io/nvidia/tritonserver:24.01-py3"
echo "==> Pulling $TRITON_IMAGE"
docker pull "$TRITON_IMAGE"

# ---------- 4. Lay out the model repository --------------------------
WORK_DIR=/opt/triton
mkdir -p "$WORK_DIR/model_repository/resnet50/1"
cd "$WORK_DIR"

# Inline the export script (kept in lockstep with model/export_model.py
# in the team repo — opset 18, dynamo=False, single-file ONNX).
cat > export_model.py <<'PYEOF'
import hashlib, os, sys
import numpy as np
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights

OUTPUT_PATH = "model_repository/resnet50/1/model.onnx"
INPUT_SHAPE = (1, 3, 224, 224)
OPSET_VERSION = 18
INPUT_NAME = "input"
OUTPUT_NAME = "output"

for path in [OUTPUT_PATH, OUTPUT_PATH + ".data"]:
    if os.path.exists(path):
        os.remove(path)

print("Loading ResNet-50 (IMAGENET1K_V2)...")
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
dummy = torch.randn(*INPUT_SHAPE, dtype=torch.float32)

print(f"Exporting → {OUTPUT_PATH} (opset {OPSET_VERSION}, legacy exporter)")
torch.onnx.export(
    model, dummy, OUTPUT_PATH,
    export_params=True, opset_version=OPSET_VERSION,
    do_constant_folding=True,
    input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
    dynamo=False,
)

if os.path.exists(OUTPUT_PATH + ".data"):
    print("ERROR: external data file created — aborting"); sys.exit(1)

import onnx, onnxruntime as ort
onnx.checker.check_model(onnx.load(OUTPUT_PATH))
with torch.no_grad():
    torch_out = model(dummy).numpy()
ort_out = ort.InferenceSession(OUTPUT_PATH, providers=["CPUExecutionProvider"]) \
            .run([OUTPUT_NAME], {INPUT_NAME: dummy.numpy()})[0]
diff = float(np.max(np.abs(torch_out - ort_out)))
print(f"Numerical max-diff: {diff:.2e}")
assert ort_out.shape == (1, 1000)

h = hashlib.sha256()
with open(OUTPUT_PATH, "rb") as f:
    for chunk in iter(lambda: f.read(8192), b""):
        h.update(chunk)
print(f"SHA-256: {h.hexdigest()}")
PYEOF

echo "==> Exporting ResNet-50 → ONNX"
python3 export_model.py

# ---------- 5. Triton model config (matches team's config.pbtxt) -----
cat > model_repository/resnet50/config.pbtxt <<'CFGEOF'
name: "resnet50"
platform: "onnxruntime_onnx"
max_batch_size: 0

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 1, 3, 224, 224 ]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1, 1000 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_AUTO
  }
]
CFGEOF

# ---------- 6. Start Triton -----------------------------------------
docker rm -f triton 2>/dev/null || true
docker run -d \
  --name triton \
  --restart=unless-stopped \
  -p 8000:8000 -p 8002:8002 \
  -v "$WORK_DIR/model_repository:/models:ro" \
  "$TRITON_IMAGE" \
  tritonserver \
    --model-repository=/models \
    --strict-model-config=true \
    --allow-http=true \
    --allow-metrics=true \
    --log-verbose=0

# ---------- 7. Wait for readiness -----------------------------------
echo "==> Waiting for Triton readiness..."
for i in $(seq 1 60); do
  if curl -fsS http://localhost:8000/v2/health/ready >/dev/null 2>&1; then
    echo "==> Triton READY after ${i}s"
    break
  fi
  sleep 1
done

if ! curl -fsS http://localhost:8000/v2/health/ready >/dev/null 2>&1; then
  echo "==> Triton FAILED to become ready in 60s"
  docker logs triton --tail 100
  exit 1
fi

echo "==> [$(date -Is)] Setup complete"
docker ps
