#!/bin/bash
# confidential/cloud_nat.sh
# Creates a Cloud Router + Cloud NAT for the default VPC in us-central1.
#
# Why this exists:
#   The applied-ml-cloud project's org policy `compute.vmExternalIpAccess`
#   forbids external IPs on VMs. Without external IPs and without NAT,
#   the VMs cannot reach apt-get repos, PyPI, or NGC. Private Google
#   Access only helps for Google-hosted endpoints, not Ubuntu's mirrors.
#   Cloud NAT solves egress without giving any VM an external IP.
#
# Idempotent — succeeds whether or not the router/NAT already exist.
#
# Usage: bash confidential/cloud_nat.sh [project-id]

set -e

PROJECT_ID="${1:-$(gcloud config get-value project)}"
REGION="us-central1"
ROUTER="nat-router"
NAT="nat-config"
NETWORK="default"

echo "Project: $PROJECT_ID"
echo "Region:  $REGION"
echo ""

# ---------- Cloud Router ------------------------------------------------
if gcloud compute routers describe "$ROUTER" \
     --project="$PROJECT_ID" --region="$REGION" >/dev/null 2>&1; then
  echo "Router '$ROUTER': already exists"
else
  echo "Router '$ROUTER': creating"
  gcloud compute routers create "$ROUTER" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --network="$NETWORK" \
    --description="NAT egress for benchmarking VMs (no external IPs)"
fi

# ---------- Cloud NAT ---------------------------------------------------
if gcloud compute routers nats describe "$NAT" \
     --router="$ROUTER" --project="$PROJECT_ID" --region="$REGION" \
     >/dev/null 2>&1; then
  echo "NAT '$NAT':       already exists"
else
  echo "NAT '$NAT':       creating"
  gcloud compute routers nats create "$NAT" \
    --router="$ROUTER" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --nat-all-subnet-ip-ranges \
    --auto-allocate-nat-external-ips
fi

echo ""
echo "Done. VMs in the default VPC in $REGION can now reach the public"
echo "internet outbound. No VM has been given an external IP."
