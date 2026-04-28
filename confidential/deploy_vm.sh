#!/bin/bash
# confidential/deploy_vm.sh
# Provisions one GCE VM for the confidential-computing slice. Idempotent —
# deletes any existing VM with the same name before creating.
#
# Usage:
#   bash confidential/deploy_vm.sh <project-id> <variant>
#
# variant: "standard"     → no confidential computing
#          "confidential" → AMD SEV enabled
#
# Both variants use n2d-standard-4 (AMD EPYC) so the matched-pair
# comparison isolates SEV overhead from machine-type differences.
# (The team protocol originally said n2-standard-4 — that is wrong;
#  n2-* is Intel and does not support AMD SEV.)

set -euo pipefail

PROJECT_ID="${1:-}"
VARIANT="${2:-}"

if [[ -z "$PROJECT_ID" || -z "$VARIANT" ]]; then
  echo "Usage: bash confidential/deploy_vm.sh <project-id> <standard|confidential>" >&2
  exit 1
fi

case "$VARIANT" in
  standard)     VM_NAME="triton-cpu-standard"     ;;
  confidential) VM_NAME="triton-cpu-confidential" ;;
  *) echo "variant must be 'standard' or 'confidential'" >&2; exit 1 ;;
esac

ZONE="us-central1-a"
MACHINE_TYPE="n2d-standard-4"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
DISK_SIZE="50GB"
TAG="triton-server"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARTUP_SCRIPT="${SCRIPT_DIR}/setup_vm.sh"

if [[ ! -f "$STARTUP_SCRIPT" ]]; then
  echo "ERROR: $STARTUP_SCRIPT not found" >&2
  exit 1
fi

echo "================================================="
echo "Project:      $PROJECT_ID"
echo "Variant:      $VARIANT"
echo "VM name:      $VM_NAME"
echo "Machine type: $MACHINE_TYPE"
echo "Zone:         $ZONE"
echo "Image:        $IMAGE_FAMILY ($IMAGE_PROJECT)"
echo "================================================="

# ---------- Idempotency: delete existing VM with same name --------------
if gcloud compute instances describe "$VM_NAME" \
     --project="$PROJECT_ID" --zone="$ZONE" >/dev/null 2>&1; then
  echo ""
  echo "Existing VM '$VM_NAME' found — deleting first."
  gcloud compute instances delete "$VM_NAME" \
    --project="$PROJECT_ID" --zone="$ZONE" --quiet
fi

# ---------- Build common args -------------------------------------------
COMMON_ARGS=(
  --project="$PROJECT_ID"
  --zone="$ZONE"
  --machine-type="$MACHINE_TYPE"
  --image-family="$IMAGE_FAMILY"
  --image-project="$IMAGE_PROJECT"
  --boot-disk-size="$DISK_SIZE"
  --boot-disk-type="pd-balanced"
  --tags="$TAG"
  --metadata-from-file=startup-script="$STARTUP_SCRIPT"
  --scopes=cloud-platform
  # Org policy on applied-ml-cloud forbids external IPs on VMs.
  --no-address
)

# ---------- Confidential-only flags -------------------------------------
if [[ "$VARIANT" == "confidential" ]]; then
  # AMD SEV requires terminate-on-maintenance (no live migration)
  COMMON_ARGS+=(
    --confidential-compute-type=SEV
    --maintenance-policy=TERMINATE
  )
fi

echo ""
echo "Creating VM..."
gcloud compute instances create "$VM_NAME" "${COMMON_ARGS[@]}"

# ---------- Print internal IP for the runner VM to use ------------------
INTERNAL_IP=$(gcloud compute instances describe "$VM_NAME" \
  --project="$PROJECT_ID" --zone="$ZONE" \
  --format='get(networkInterfaces[0].networkIP)')

echo ""
echo "================================================="
echo "VM created: $VM_NAME"
echo "Internal IP: $INTERNAL_IP  (no external IP — org policy)"
echo ""
echo "Startup script will install Docker + Triton + ResNet-50."
echo "Expected ready time: ~5-7 minutes."
echo ""
echo "Watch startup progress (works without external IP):"
echo "  gcloud compute instances get-serial-port-output $VM_NAME --zone $ZONE | tail -50"
echo ""
echo "SSH via IAP (no external IP needed; firewall.sh must have been run):"
echo "  gcloud compute ssh $VM_NAME --zone $ZONE --tunnel-through-iap"
echo ""
echo "Smoke test must run from inside the VPC (e.g., the harness-runner VM):"
echo "  curl -fsS http://${INTERNAL_IP}:8000/v2/health/ready && echo OK"
echo ""
echo "Run benchmarks from the harness-runner VM:"
case "$VARIANT" in
  standard)
    echo "  bash harness/run_all.sh http://${INTERNAL_IP}:8000 cpu" ;;
  confidential)
    echo "  bash harness/run_all.sh http://${INTERNAL_IP}:8000 confidential" ;;
esac
echo "================================================="
