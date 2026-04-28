#!/bin/bash
# confidential/firewall.sh
# Sets up two firewall rules in the default VPC:
#   1. allow-iap-ssh        — IAP source range → TCP 22 on triton-server tag
#                             (lets Eric SSH into VMs that have no external IP)
#   2. allow-triton-internal — VPC internal → TCP 8000 on triton-server tag
#                             (lets the harness-runner VM reach Triton)
#
# (Rule #1 was previously TCP 8000 from 0.0.0.0/0; that's not usable here
#  because the project's org policy forbids external IPs on VMs.)
#
# Idempotent — succeeds whether or not the rules already exist.
#
# Usage: bash confidential/firewall.sh [project-id]

set -e

PROJECT_ID="${1:-$(gcloud config get-value project)}"

echo "Project: $PROJECT_ID"
echo ""

ensure_rule () {
  local NAME="$1" ; shift
  if gcloud compute firewall-rules describe "$NAME" --project="$PROJECT_ID" >/dev/null 2>&1; then
    echo "  $NAME: already exists"
  else
    echo "  $NAME: creating"
    gcloud compute firewall-rules create "$NAME" --project="$PROJECT_ID" "$@"
  fi
}

# IAP TCP forwarding source range — fixed by Google
# https://cloud.google.com/iap/docs/using-tcp-forwarding
ensure_rule allow-iap-ssh \
  --network=default \
  --direction=INGRESS \
  --action=ALLOW \
  --rules=tcp:22 \
  --source-ranges=35.235.240.0/20 \
  --target-tags=triton-server \
  --description="IAP SSH to Triton VMs (no external IP)"

# Internal VPC traffic — let the runner VM (and any VM with the tag) hit Triton on 8000
ensure_rule allow-triton-internal \
  --network=default \
  --direction=INGRESS \
  --action=ALLOW \
  --rules=tcp:8000 \
  --source-ranges=10.0.0.0/8 \
  --target-tags=triton-server \
  --description="Internal VPC access to Triton on TCP 8000"

echo ""
echo "Done."
