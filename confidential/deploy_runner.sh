#!/bin/bash
# confidential/deploy_runner.sh
# Provisions a small VM in the same VPC that runs the benchmarking
# harness against the Triton VMs over internal IPs. This is the
# work-around for `compute.vmExternalIpAccess` org policy: no VM
# (Triton or runner) has an external IP, but the runner can reach
# Triton via the default VPC's internal routing.
#
# Why a runner VM and not IAP-tunnel-from-laptop?
#   IAP TCP forwarding adds 5–15ms per request and multiplexes
#   "concurrent clients" through Google's edge — which would
#   contaminate the latency/concurrency measurements. A runner VM
#   in the same VPC gives clean internal-IP networking.
#
# Usage:
#   bash confidential/deploy_runner.sh <project-id>

set -euo pipefail

PROJECT_ID="${1:-}"
if [[ -z "$PROJECT_ID" ]]; then
  echo "Usage: bash confidential/deploy_runner.sh <project-id>" >&2
  exit 1
fi

VM_NAME="harness-runner"
ZONE="us-central1-a"
MACHINE_TYPE="e2-standard-2"     # 2 vCPU, 8 GB RAM — plenty for the harness
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
DISK_SIZE="20GB"

# Inline startup script — installs Python and the harness's deps.
# The actual harness code is uploaded later via `gcloud compute scp`.
read -r -d '' STARTUP <<'STARTUPEOF' || true
#!/bin/bash
set -e
exec > >(tee -a /var/log/runner-setup.log) 2>&1
echo "==> [$(date -Is)] runner setup"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates curl python3-pip python3-venv python-is-python3 git
python3 -m pip install --no-cache-dir --upgrade pip
python3 -m pip install --no-cache-dir \
  "tritonclient[http]==2.43.0" numpy Pillow
echo "==> runner ready"
STARTUPEOF

echo "================================================="
echo "Project:      $PROJECT_ID"
echo "VM name:      $VM_NAME (harness runner)"
echo "Machine type: $MACHINE_TYPE"
echo "Zone:         $ZONE"
echo "================================================="

# Idempotency
if gcloud compute instances describe "$VM_NAME" \
     --project="$PROJECT_ID" --zone="$ZONE" >/dev/null 2>&1; then
  echo ""
  echo "Existing VM '$VM_NAME' found — deleting first."
  gcloud compute instances delete "$VM_NAME" \
    --project="$PROJECT_ID" --zone="$ZONE" --quiet
fi

# Use a temp file for the startup script so we can pass it via --metadata-from-file
TMP_STARTUP="$(mktemp)"
trap 'rm -f "$TMP_STARTUP"' EXIT
printf '%s\n' "$STARTUP" > "$TMP_STARTUP"

echo ""
echo "Creating runner VM..."
gcloud compute instances create "$VM_NAME" \
  --project="$PROJECT_ID" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="$DISK_SIZE" \
  --boot-disk-type="pd-balanced" \
  --tags="triton-server" \
  --metadata-from-file=startup-script="$TMP_STARTUP" \
  --scopes=cloud-platform \
  --no-address

INTERNAL_IP=$(gcloud compute instances describe "$VM_NAME" \
  --project="$PROJECT_ID" --zone="$ZONE" \
  --format='get(networkInterfaces[0].networkIP)')

echo ""
echo "================================================="
echo "Runner VM created: $VM_NAME"
echo "Internal IP:       $INTERNAL_IP"
echo ""
echo "Wait ~2 minutes for the runner's startup script to install"
echo "Python deps, then SSH in and upload the harness:"
echo ""
echo "  # 1. SSH in (via IAP)"
echo "  gcloud compute ssh $VM_NAME --zone $ZONE --tunnel-through-iap"
echo ""
echo "  # 2. From your laptop, push the harness + model files:"
echo "  gcloud compute scp --tunnel-through-iap --zone $ZONE \\"
echo "    --recurse harness/ $VM_NAME:~"
echo ""
echo "  # 3. From inside the runner, run benchmarks:"
echo "  STD_IP=<from deploy_vm.sh standard>"
echo "  bash harness/run_all.sh http://\$STD_IP:8000 cpu"
echo "================================================="
