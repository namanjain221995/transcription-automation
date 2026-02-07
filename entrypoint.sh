#!/usr/bin/env bash
set -euo pipefail

echo "== Python =="
python --version

echo "== Running pipeline scripts =="
python test9.py

echo "== All scripts finished successfully =="

# Only stop EC2 if enabled
if [ "${STOP_EC2_ON_SUCCESS:-0}" != "1" ]; then
  echo "STOP_EC2_ON_SUCCESS != 1, so NOT stopping EC2."
  exit 0
fi

echo "== Stopping this EC2 instance (via IMDSv2) =="

# Get IMDSv2 token
TOKEN="$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" || true)"

if [ -z "$TOKEN" ]; then
  echo "ERROR: Could not get IMDSv2 token. Is IMDS enabled?"
  exit 1
fi

INSTANCE_ID="$(curl -sS -H "X-aws-ec2-metadata-token: $TOKEN" \
  "http://169.254.169.254/latest/meta-data/instance-id" || true)"

REGION="$(curl -sS -H "X-aws-ec2-metadata-token: $TOKEN" \
  "http://169.254.169.254/latest/meta-data/placement/region" || true)"

if [ -z "$INSTANCE_ID" ] || [ -z "$REGION" ]; then
  echo "ERROR: Could not read instance-id or region from metadata."
  echo "INSTANCE_ID='$INSTANCE_ID' REGION='$REGION'"
  exit 1
fi

echo "Instance: $INSTANCE_ID | Region: $REGION"
echo "Calling: aws ec2 stop-instances ..."
aws ec2 stop-instances --region "$REGION" --instance-ids "$INSTANCE_ID" >/dev/null

echo "Stop request sent. Done."
