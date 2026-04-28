#!/bin/bash
# confidential/teardown.sh
# Deletes the three VMs (standard, confidential, runner) and the
# slice's firewall rules once experiments are complete. Run this
# AFTER the benchmark CSVs are committed — VMs bill while running.
#
# Usage:
#   bash confidential/teardown.sh <project-id> [--keep-firewall]

set -e

PROJECT_ID="${1:-$(gcloud config get-value project)}"
KEEP_FW="${2:-}"
ZONE="us-central1-a"

echo "Project: $PROJECT_ID"
echo ""

for VM in triton-cpu-standard triton-cpu-confidential harness-runner; do
  if gcloud compute instances describe "$VM" \
       --project="$PROJECT_ID" --zone="$ZONE" >/dev/null 2>&1; then
    echo "Deleting $VM..."
    gcloud compute instances delete "$VM" \
      --project="$PROJECT_ID" --zone="$ZONE" --quiet
  else
    echo "$VM: not present, skipping"
  fi
done

if [[ "$KEEP_FW" != "--keep-firewall" ]]; then
  for RULE in allow-iap-ssh allow-triton-internal allow-triton-http; do
    if gcloud compute firewall-rules describe "$RULE" \
         --project="$PROJECT_ID" >/dev/null 2>&1; then
      echo "Deleting firewall rule $RULE..."
      gcloud compute firewall-rules delete "$RULE" \
        --project="$PROJECT_ID" --quiet
    fi
  done

  # Cloud NAT (and the router that hosts it). NAT bills hourly even
  # idle, so always tear it down with the slice unless --keep-firewall.
  if gcloud compute routers nats describe nat-config \
       --router=nat-router --project="$PROJECT_ID" \
       --region=us-central1 >/dev/null 2>&1; then
    echo "Deleting Cloud NAT nat-config..."
    gcloud compute routers nats delete nat-config \
      --router=nat-router --project="$PROJECT_ID" \
      --region=us-central1 --quiet
  fi
  if gcloud compute routers describe nat-router \
       --project="$PROJECT_ID" --region=us-central1 >/dev/null 2>&1; then
    echo "Deleting Cloud Router nat-router..."
    gcloud compute routers delete nat-router \
      --project="$PROJECT_ID" --region=us-central1 --quiet
  fi
fi

echo ""
echo "Teardown complete."
