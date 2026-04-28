#!/usr/bin/env bash
#
# Deploy the Cloud Function (2nd gen) for the serverless slice of Project 2.
#
# Defaults match the team-aligned config: 4 vCPU / 8 GB / 540s timeout /
# Python 3.11, with autoscaling pinned at 1 instance to remove cold-start
# variance from the warm benchmark.
#
# Usage:
#   bash cloudfunction/deploy.sh                  # uses default project
#   bash cloudfunction/deploy.sh <project-id>     # override project
#
# After deploy, prints the HTTPS URL to use as the harness --endpoint.

set -euo pipefail

# Resolve to the directory containing this script — works whether invoked
# from the repo root or from within cloudfunction/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROJECT_ID="${1:-ancient-acumen-486002-j4}"
REGION="us-central1"
FUNCTION_NAME="triton-resnet50-cf"
RUNTIME="python311"
ENTRY_POINT="triton_handler"

# Pre-flight: verify the model artifact is present. Without it, the deployed
# function fails at module import (ort.InferenceSession can't open the file).
if [[ ! -f "resnet50.onnx" ]]; then
  echo "ERROR: cloudfunction/resnet50.onnx is missing." >&2
  echo "Run: python model/export_model.py && cp resnet50.onnx cloudfunction/" >&2
  exit 1
fi

echo "Deploying $FUNCTION_NAME"
echo "  project:  $PROJECT_ID"
echo "  region:   $REGION"
echo "  runtime:  $RUNTIME"
echo "  entry:    $ENTRY_POINT"
echo "  cpu/mem:  4 vCPU / 8 GiB"
echo "  scaling:  min=1, max=1"
echo "  timeout:  540s"
echo

gcloud functions deploy "$FUNCTION_NAME" \
  --gen2 \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --runtime="$RUNTIME" \
  --source=. \
  --entry-point="$ENTRY_POINT" \
  --trigger-http \
  --allow-unauthenticated \
  --memory=8Gi \
  --cpu=4 \
  --timeout=540s \
  --min-instances=1 \
  --max-instances=1 \
  --concurrency=100

echo
# Fetch the URL via the REST API rather than `gcloud functions describe`.
# gcloud's response parser hits a protobuf incompatibility on some macOS
# installs (`AttributeError: module 'google._upb._message' has no attribute
# 'MessageMapContainer'`) — REST + access token sidesteps it entirely.
TOKEN="$(gcloud auth print-access-token)"
URL="$(curl -sS -H "Authorization: Bearer $TOKEN" \
  "https://cloudfunctions.googleapis.com/v2/projects/$PROJECT_ID/locations/$REGION/functions/$FUNCTION_NAME" \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['serviceConfig']['uri'])")"

echo "Endpoint: $URL"
echo
echo "Next step — smoke test:"
echo "  python cloudfunction/test_inference.py --endpoint $URL"
echo
echo "Then run the benchmark:"
echo "  bash harness/run_all.sh $URL cloudfunction"
