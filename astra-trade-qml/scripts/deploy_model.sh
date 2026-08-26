#!/usr/bin/env bash
# Syncs the latest trained model to the S3 bucket configured in config.yaml
# (infrastructure.model_sync), so the Hetzner paper-trading host can pick it up.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BUCKET="${S3_MODEL_BUCKET:-astra-trade-models}"
REGION="${AWS_REGION:-ap-south-1}"
MODEL_DIR="${MODEL_DIR:-models/latest}"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "No model found at $MODEL_DIR — run scripts/run_training.sh first." >&2
  exit 1
fi

aws s3 sync "$MODEL_DIR" "s3://$BUCKET/latest" --region "$REGION"
echo "Model synced to s3://$BUCKET/latest"
