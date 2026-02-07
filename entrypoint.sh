#!/usr/bin/env bash
set -euo pipefail

echo "== Running scripts =="
python test9.py

echo "== Scripts OK. Stopping EC2 instance now. =="

# ---- IMDSv2 token ----
TOKEN="$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")"

if [ -z "$TOKEN" ]; then
  echo "ERROR: Failed to get IMDSv2 token."
  exit 1
fi

# ---- instance-id + region ----
INSTANCE_ID="$(curl -sS -H "X-aws-ec2-metadata-token: $TOKEN" \
  "http://169.254.169.254/latest/meta-data/instance-id")"

REGION="$(curl -sS -H "X-aws-ec2-metadata-token: $TOKEN" \
  "http://169.254.169.254/latest/meta-data/placement/region")"

if [ -z "$INSTANCE_ID" ] || [ -z "$REGION" ]; then
  echo "ERROR: Could not read instance-id/region from metadata."
  echo "INSTANCE_ID='$INSTANCE_ID' REGION='$REGION'"
  exit 1
fi

echo "Instance: $INSTANCE_ID"
echo "Region:   $REGION"

# ---- sanity check creds ----
echo "== Checking AWS creds from instance role =="
aws sts get-caller-identity --region "$REGION"

echo "== Calling stop-instances =="
aws ec2 stop-instances --region "$REGION" --instance-ids "$INSTANCE_ID"

echo "Stop requested. Bye."
exit 0
