#!/usr/bin/env bash
# cloudrun/benchmark_from_gce.sh
#
# Benchmark the deployed Cloud Run Triton service from a GCE VM client
# in the same region (us-central1-a). Eliminates the variable home/ISP/edge
# network latency present when benchmarking from a local laptop, so the
# recorded latency_ms reflects pure server-side cost.
#
# This project enforces the org policy constraints/compute.vmExternalIpAccess,
# so the VM is created with NO external IP. SSH happens via IAP tunneling, and
# outbound HTTPS to the Cloud Run endpoint goes through Cloud NAT.
#
# Prerequisites (must be true BEFORE running this script):
#   1. Cloud Run Triton service deployed and reachable. See cloudrun/CLAUDE.md
#      for the build/push/deploy procedure.
#   2. gcloud CLI installed and authenticated (gcloud auth login).
#   3. Cloud NAT configured in us-central1 (one-time setup; see error message
#      below if missing).
#   4. Run from the repo root.
#
# Usage:
#   bash cloudrun/benchmark_from_gce.sh <endpoint-url> [--keep-instance] [--instance-name <name>]
#
# Output:
#   results/cloudrun/from_gce/results_cloudrun_<C>_<R>.csv  (20 files expected)

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT="ancient-acumen-486002-j4"
# Tried in order; first zone with capacity wins. All in us-central1 to keep
# intra-region RTT to the Cloud Run service.
ZONES_PREFERRED=("us-central1-a" "us-central1-b" "us-central1-c" "us-central1-f")
ZONE="${ZONES_PREFERRED[0]}"   # placeholder; set for real after instance creation
# Preferred machine types, tried in order. n2 is the methodologically cleaner
# choice (predictable hardware, low jitter); e2 is a cost-and-availability
# fallback when n2 capacity is tight across all zones.
MACHINE_TYPES_PREFERRED=("n2-standard-4" "e2-standard-4")
MACHINE_TYPE="${MACHINE_TYPES_PREFERRED[0]}"   # placeholder; set for real after creation
# Approximate us-central1 list price ($/hr) per machine type, for the cost summary.
# (bash 3.2 ships on macOS, so avoid associative arrays — use a case lookup.)
hourly_rate_for() {
  case "$1" in
    n2-standard-4) echo "0.19" ;;
    e2-standard-4) echo "0.13" ;;
    *)             echo "0.20" ;;  # conservative default
  esac
}
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
BOOT_DISK_SIZE="20GB"
BOOT_DISK_TYPE="pd-balanced"
CONFIG_NAME="cloudrun"
RESULTS_DIR="results/cloudrun/from_gce"

# ── Argument parsing ─────────────────────────────────────────────────────────
ENDPOINT=""
KEEP_INSTANCE=0
INSTANCE_NAME="benchmark-client-$(date +%s)"
CONCURRENCY_ONLY=""   # empty = full 20-run protocol; set to e.g. 100 for targeted re-runs

usage() {
  cat <<EOF
Usage: bash cloudrun/benchmark_from_gce.sh <endpoint-url> [options]

  <endpoint-url>          Cloud Run endpoint, e.g. https://triton-resnet50-xxxxx-uc.a.run.app
  --keep-instance         Skip VM deletion at end (for iterative debugging)
  --instance-name <name>  Override auto-generated VM name (default: benchmark-client-<timestamp>)
  --concurrency-only <N>  Run only 5 rounds of concurrency N (1|10|50|100) instead of
                          the full 20-run protocol. Uses 60s inter-run pause so Triton
                          can recover between high-concurrency runs.
EOF
  exit 1
}

if [ $# -lt 1 ]; then
  usage
fi

ENDPOINT="$1"
shift

while [ $# -gt 0 ]; do
  case "$1" in
    --keep-instance)
      KEEP_INSTANCE=1
      shift
      ;;
    --instance-name)
      if [ $# -lt 2 ]; then
        echo "ERROR: --instance-name requires a value"
        usage
      fi
      INSTANCE_NAME="$2"
      shift 2
      ;;
    --concurrency-only)
      if [ $# -lt 2 ]; then
        echo "ERROR: --concurrency-only requires a value (1, 10, 50, or 100)"
        usage
      fi
      CONCURRENCY_ONLY="$2"
      case "$CONCURRENCY_ONLY" in
        1|10|50|100) ;;
        *) echo "ERROR: --concurrency-only must be one of: 1 10 50 100"; exit 1 ;;
      esac
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "ERROR: unknown argument: $1"
      usage
      ;;
  esac
done

# ── Banner ───────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  GCE Benchmark Client — Cloud Run"
echo "════════════════════════════════════════════════════════════"
echo "  Endpoint:       $ENDPOINT"
echo "  Instance:       $INSTANCE_NAME"
echo "  Zones (try):    ${ZONES_PREFERRED[*]}"
echo "  Machines (try): ${MACHINE_TYPES_PREFERRED[*]}"
echo "  External IP:    no (IAP tunneling, Cloud NAT for egress)"
echo "  Auto-delete:    $([ $KEEP_INSTANCE -eq 0 ] && echo yes || echo no)"
echo "  Mode:           $([ -n "$CONCURRENCY_ONLY" ] && echo "concurrency-only ($CONCURRENCY_ONLY), 5 runs, 60s pause" || echo "full protocol (1 10 50 100 × 5 runs)")"
echo "════════════════════════════════════════════════════════════"
echo ""

# ── Preflight checks ─────────────────────────────────────────────────────────
echo "→ Preflight checks..."

# Repo-root check (we expect harness/ and cloudrun/ siblings)
if [ ! -d "harness" ] || [ ! -d "cloudrun" ]; then
  echo "  ERROR: must run from the repo root (where harness/ and cloudrun/ live)."
  exit 1
fi

# Required local files
for f in harness/harness.py harness/requirements.txt harness/run_all.sh; do
  if [ ! -f "$f" ]; then
    echo "  ERROR: missing required file: $f"
    exit 1
  fi
done

# gcloud installed?
if ! command -v gcloud >/dev/null 2>&1; then
  echo "  ERROR: gcloud CLI not found in PATH."
  echo "  Install: https://cloud.google.com/sdk/docs/install"
  exit 1
fi

# gcloud authenticated?
ACCOUNT=$(gcloud config get-value account 2>/dev/null || true)
if [ -z "$ACCOUNT" ] || [ "$ACCOUNT" = "(unset)" ]; then
  echo "  ERROR: gcloud not authenticated. Run: gcloud auth login"
  exit 1
fi

# Endpoint URL shape
if [[ ! "$ENDPOINT" =~ ^https?:// ]]; then
  echo "  ERROR: endpoint must start with http:// or https://"
  exit 1
fi

# Probe endpoint health (Triton's /v2/health/ready returns 200 when ready)
echo "  Probing $ENDPOINT/v2/health/ready ..."
if ! curl -sf -o /dev/null --max-time 10 "$ENDPOINT/v2/health/ready"; then
  echo "  ERROR: endpoint did not respond healthy on /v2/health/ready."
  echo "  Verify the Cloud Run service is deployed and reachable."
  exit 1
fi

# IAP API enabled? (idempotent — returns immediately if already enabled.)
echo "  Ensuring iap.googleapis.com is enabled..."
if ! gcloud services enable iap.googleapis.com --project="$PROJECT" --quiet 2>/dev/null; then
  echo "  ERROR: failed to enable iap.googleapis.com. Verify your gcloud user"
  echo "  has serviceusage.services.enable permission on project $PROJECT."
  exit 1
fi

# Cloud NAT in us-central1? (without it, the no-external-IP VM cannot reach
# the Cloud Run public endpoint and the benchmark will hang/fail later.)
echo "  Verifying Cloud NAT exists in us-central1..."
NAT_FOUND=0
ROUTERS=$(gcloud compute routers list --filter="region:us-central1" \
            --project="$PROJECT" --format="value(name)" 2>/dev/null || true)
for router in $ROUTERS; do
  NATS=$(gcloud compute routers nats list --router="$router" \
           --region=us-central1 --project="$PROJECT" \
           --format="value(name)" 2>/dev/null || true)
  if [ -n "$NATS" ]; then
    NAT_FOUND=1
    break
  fi
done
if [ $NAT_FOUND -eq 0 ]; then
  cat <<EOF
  ERROR: no Cloud NAT configured in us-central1.

  This project enforces constraints/compute.vmExternalIpAccess, so the
  benchmark VM has no external IP. It needs Cloud NAT for outbound HTTPS
  to the Cloud Run endpoint.

  One-time setup (run once, then re-run this script):

    gcloud compute routers create benchmark-router \\
      --network=default --region=us-central1 \\
      --project=$PROJECT

    gcloud compute routers nats create benchmark-nat \\
      --router=benchmark-router --region=us-central1 \\
      --auto-allocate-nat-external-ips \\
      --nat-all-subnet-ip-ranges \\
      --project=$PROJECT

  Cost: ~\$0.045/hour while the NAT exists, plus per-GB egress (negligible
  for benchmarking).
EOF
  exit 1
fi

echo "  All preflight checks passed."
echo ""

# ── Phase 1: Create VM ───────────────────────────────────────────────────────
START_TIME=$(date +%s)

echo "→ [1/8] Creating GCE instance: $INSTANCE_NAME"

# Idempotency: if this name already exists (e.g. from a prior --keep-instance
# run), delete it first so the create doesn't fail.
EXISTING_ZONE=$(gcloud compute instances list \
  --filter="name=$INSTANCE_NAME AND zone:us-central1" \
  --project="$PROJECT" \
  --format="value(zone)" 2>/dev/null | head -1 || true)
if [ -n "$EXISTING_ZONE" ]; then
  echo "  Found existing instance $INSTANCE_NAME in $EXISTING_ZONE — deleting..."
  gcloud compute instances delete "$INSTANCE_NAME" \
    --zone="$EXISTING_ZONE" --project="$PROJECT" --quiet
  echo "  Deleted."
fi

CREATED_ZONE=""
CREATED_MACHINE_TYPE=""
CREATE_ERR=$(mktemp)
for candidate_machine in "${MACHINE_TYPES_PREFERRED[@]}"; do
  if [ -n "$CREATED_ZONE" ]; then
    break
  fi
  if [ "$candidate_machine" != "${MACHINE_TYPES_PREFERRED[0]}" ]; then
    echo ""
    echo "  ⚠️  Falling back to $candidate_machine (capacity unavailable for prior types)."
    echo "     Note this in your run log: e2 has slightly higher client-side jitter"
    echo "     than n2 — p95/p99 tails on these CSVs may be a few ms wider."
    echo ""
  fi
  for candidate_zone in "${ZONES_PREFERRED[@]}"; do
    echo "  Trying $candidate_machine in $candidate_zone..."
    if gcloud compute instances create "$INSTANCE_NAME" \
         --project="$PROJECT" \
         --zone="$candidate_zone" \
         --machine-type="$candidate_machine" \
         --image-family="$IMAGE_FAMILY" \
         --image-project="$IMAGE_PROJECT" \
         --boot-disk-size="$BOOT_DISK_SIZE" \
         --boot-disk-type="$BOOT_DISK_TYPE" \
         --scopes=cloud-platform \
         --no-address \
         --quiet >/dev/null 2>"$CREATE_ERR"; then
      CREATED_ZONE="$candidate_zone"
      CREATED_MACHINE_TYPE="$candidate_machine"
      echo "  Created: $candidate_machine in $candidate_zone (no external IP)."
      break
    fi
    # Detect the transient capacity error vs. anything else
    if grep -q "ZONE_RESOURCE_POOL_EXHAUSTED\|does not have enough resources" "$CREATE_ERR"; then
      echo "    $candidate_zone: out of $candidate_machine capacity."
      continue
    fi
    # A non-capacity error — surface it and stop (don't waste time iterating)
    echo "  Instance create failed with a non-capacity error:"
    cat "$CREATE_ERR"
    rm -f "$CREATE_ERR"
    exit 1
  done
done
rm -f "$CREATE_ERR"
if [ -z "$CREATED_ZONE" ]; then
  echo "  ERROR: every machine type was unavailable in every us-central1 zone tried."
  echo "  Tried: ${MACHINE_TYPES_PREFERRED[*]} × ${ZONES_PREFERRED[*]}"
  echo "  Re-run in 5–15 min, or extend MACHINE_TYPES_PREFERRED in the script."
  exit 1
fi
ZONE="$CREATED_ZONE"
MACHINE_TYPE="$CREATED_MACHINE_TYPE"
echo ""

# Cleanup trap: if anything below fails, delete the instance (unless --keep-instance).
cleanup_on_error() {
  local exit_code=$?
  if [ $exit_code -ne 0 ] && [ $KEEP_INSTANCE -eq 0 ]; then
    echo ""
    echo "→ Script failed (exit $exit_code). Deleting instance..."
    gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet >/dev/null 2>&1 || true
  fi
  exit $exit_code
}
trap cleanup_on_error EXIT

# ── Phase 2: Wait for SSH-ready ──────────────────────────────────────────────
echo "→ [2/8] Waiting for SSH (via IAP) to become available..."
MAX_ATTEMPTS=18  # IAP can take a bit longer than direct SSH on first attempt
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
  ATTEMPT=$((ATTEMPT + 1))
  if gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --quiet \
       --tunnel-through-iap \
       --command="echo ready" \
       --ssh-flag="-o ConnectTimeout=10" \
       --ssh-flag="-o StrictHostKeyChecking=no" \
       --ssh-flag="-o ServerAliveInterval=30" \
       >/dev/null 2>&1; then
    echo "  SSH ready (attempt $ATTEMPT)."
    break
  fi
  echo "  Attempt $ATTEMPT/$MAX_ATTEMPTS — not ready yet, sleeping 10s..."
  sleep 10
done
if [ $ATTEMPT -ge $MAX_ATTEMPTS ] && \
   ! gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --quiet \
       --tunnel-through-iap \
       --command="echo ready" \
       --ssh-flag="-o ConnectTimeout=10" \
       --ssh-flag="-o StrictHostKeyChecking=no" \
       >/dev/null 2>&1; then
  echo "  ERROR: SSH did not become available after $((MAX_ATTEMPTS * 10))s."
  echo "  Common causes:"
  echo "    - Default firewall rule blocking IAP range (35.235.240.0/20)."
  echo "      Fix: gcloud compute firewall-rules create allow-iap-ssh \\"
  echo "             --network=default --direction=INGRESS \\"
  echo "             --source-ranges=35.235.240.0/20 --allow=tcp:22"
  echo "    - Your gcloud user lacks roles/iap.tunnelResourceAccessor."
  exit 1
fi
echo ""

# Common SSH flags for the long-running phases below.
SSH_FLAGS=(
  --quiet
  --tunnel-through-iap
  --ssh-flag="-o StrictHostKeyChecking=no"
  --ssh-flag="-o ServerAliveInterval=30"
  --ssh-flag="-o ServerAliveCountMax=120"  # tolerate ~60 min idle
)

# ── Phase 3: Create remote directories ───────────────────────────────────────
echo "→ [3/8] Creating remote working directories..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" "${SSH_FLAGS[@]}" \
  --command="mkdir -p ~/harness ~/results"
echo ""

# ── Phase 4: Copy harness files ──────────────────────────────────────────────
echo "→ [4/8] Copying harness files to instance..."
gcloud compute scp --zone="$ZONE" --quiet --tunnel-through-iap \
  --scp-flag="-o StrictHostKeyChecking=no" \
  harness/harness.py harness/requirements.txt harness/run_all.sh \
  "$INSTANCE_NAME:~/harness/"
echo ""

# ── Phase 5: Install Python venv + dependencies on VM ────────────────────────
echo "→ [5/8] Installing Python and harness dependencies on VM..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" "${SSH_FLAGS[@]}" --command="
  set -e
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-pip python3-venv python-is-python3 ca-certificates
  sudo update-ca-certificates
  python3 -m venv ~/venv
  source ~/venv/bin/activate
  pip install --quiet --upgrade pip
  pip install --quiet certifi
  pip install --quiet -r ~/harness/requirements.txt
  echo '  setup complete.'
"
echo ""

# ── Phase 6: Run the benchmark protocol ──────────────────────────────────────
if [ -n "$CONCURRENCY_ONLY" ]; then
  echo "→ [6/8] Running concurrency=$CONCURRENCY_ONLY only (5 runs, 60s pause between runs)..."
  echo "  Output streams below in real-time."
  echo "──────────────────────────────────────────────────────────────"
  gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" "${SSH_FLAGS[@]}" --command="
    set -e
    source ~/venv/bin/activate
    export SSL_CERT_FILE=\$(python3 -c 'import certifi; print(certifi.where())')
    export REQUESTS_CA_BUNDLE=\$SSL_CERT_FILE
    mkdir -p ~/results/$CONFIG_NAME
    for run in 1 2 3 4 5; do
      echo \"--- Run \$run / 5 (concurrency $CONCURRENCY_ONLY) ---\"
      python harness/harness.py \
        --endpoint '$ENDPOINT' \
        --config-name '$CONFIG_NAME' \
        --concurrency '$CONCURRENCY_ONLY' \
        --run \$run \
        --requests 200 \
        --warmup 20 \
        --output ~/results/$CONFIG_NAME/
      if [ \$run -lt 5 ]; then
        echo \"  Pausing 60s before next run...\"
        sleep 60
      fi
    done
  "
else
  echo "→ [6/8] Running full benchmark protocol on VM (this takes ~60 min)..."
  echo "  Output streams below in real-time."
  echo "──────────────────────────────────────────────────────────────"
  gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" "${SSH_FLAGS[@]}" --command="
    set -e
    source ~/venv/bin/activate
    # Point geventhttpclient's SSL context at certifi's CA bundle.
    # gevent.ssl does not always pick up the OS CA store on minimal Debian images.
    export SSL_CERT_FILE=\$(python3 -c 'import certifi; print(certifi.where())')
    export REQUESTS_CA_BUNDLE=\$SSL_CERT_FILE
    cd ~ && bash harness/run_all.sh '$ENDPOINT' '$CONFIG_NAME'
  "
fi
echo "──────────────────────────────────────────────────────────────"
echo ""

# ── Phase 7: Pull results back ───────────────────────────────────────────────
echo "→ [7/8] Pulling result CSVs back to local..."
mkdir -p "$RESULTS_DIR"
TMP_DIR=$(mktemp -d)
gcloud compute scp --recurse --zone="$ZONE" --quiet --tunnel-through-iap \
  --scp-flag="-o StrictHostKeyChecking=no" \
  "$INSTANCE_NAME:~/results/cloudrun" \
  "$TMP_DIR/"

# Move CSVs into the flat results dir; tolerate empty case
shopt -s nullglob
csvs=("$TMP_DIR"/cloudrun/*.csv)
shopt -u nullglob
if [ ${#csvs[@]} -gt 0 ]; then
  mv "${csvs[@]}" "$RESULTS_DIR/"
fi
rm -rf "$TMP_DIR"

CSV_COUNT=$(ls "$RESULTS_DIR"/results_*.csv 2>/dev/null | wc -l | tr -d ' ')
echo "  $CSV_COUNT CSV(s) copied to $RESULTS_DIR/"
echo ""

# ── Phase 8: Cleanup ─────────────────────────────────────────────────────────
echo "→ [8/8] Cleanup..."
trap - EXIT  # disable error-trap; we're past the failure-prone phases
if [ $KEEP_INSTANCE -eq 0 ]; then
  gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --quiet >/dev/null
  echo "  Instance $INSTANCE_NAME deleted."
else
  echo "  Instance $INSTANCE_NAME left running (--keep-instance set)."
  echo "  Delete later with:"
  echo "    gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
fi
echo ""

# ── Summary ──────────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))
HOURLY_RATE_USD=$(hourly_rate_for "$MACHINE_TYPE")
COST=$(awk "BEGIN { printf \"%.2f\", $ELAPSED * $HOURLY_RATE_USD / 3600 }")

echo "════════════════════════════════════════════════════════════"
echo "  Done"
echo "════════════════════════════════════════════════════════════"
echo "  Machine used:     $MACHINE_TYPE in $ZONE"
echo "  CSVs collected:   $CSV_COUNT (expected: 20)"
echo "  Local path:       $RESULTS_DIR/"
echo "  Wall time:        ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "  Estimated cost:   ~\$$COST (\$${HOURLY_RATE_USD}/hr × ${ELAPSED_MIN}m ${ELAPSED_SEC}s)"
echo "════════════════════════════════════════════════════════════"

if [ "$CSV_COUNT" -lt 20 ]; then
  echo ""
  echo "  WARNING: fewer than 20 CSVs collected. Inspect output above for"
  echo "  failed runs (look for 'Run failed' lines from run_all.sh) and"
  echo "  consider re-running specific (concurrency, run) pairs by invoking"
  echo "  harness/harness.py directly against $ENDPOINT."
fi
