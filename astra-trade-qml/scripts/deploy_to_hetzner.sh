#!/usr/bin/env bash
# Deploys the latest trained model to the Hetzner paper-trading host and
# restarts the service so it picks it up.
#
# Expects to be run from a checkout of the `model-artifacts` branch (i.e.
# astra-trade-qml/models/latest/ present in the working directory - that's
# what the RunPod training pod pushes there; see scripts/ci/launch_training_pod.py).
#
# This only updates the model - it assumes the paper-trading docker-compose
# stack is already running on the host (docker/docker-compose.yml, set up
# once by hand per docker/README-style instructions). Redeploying the
# service's own code/image is a separate, less-frequent concern and out of
# scope here.
#
# Required env: HETZNER_HOST, HETZNER_USER, HETZNER_SSH_KEY_PATH
# Optional env: HETZNER_SSH_PORT (default 22), HETZNER_REMOTE_DIR (default
# /opt/astra-trade-qml)
set -euo pipefail

: "${HETZNER_HOST:?HETZNER_HOST is required}"
: "${HETZNER_USER:?HETZNER_USER is required}"
: "${HETZNER_SSH_KEY_PATH:?HETZNER_SSH_KEY_PATH is required}"
HETZNER_SSH_PORT="${HETZNER_SSH_PORT:-22}"
REMOTE_DIR="${HETZNER_REMOTE_DIR:-/opt/astra-trade-qml}"

MODEL_DIR="astra-trade-qml/models/latest"
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "No model found at $MODEL_DIR - nothing to deploy." >&2
  exit 1
fi

SSH_OPTS=(-i "$HETZNER_SSH_KEY_PATH" -p "$HETZNER_SSH_PORT" -o StrictHostKeyChecking=accept-new)

echo "Ensuring remote model directory exists..."
ssh "${SSH_OPTS[@]}" "${HETZNER_USER}@${HETZNER_HOST}" "mkdir -p ${REMOTE_DIR}/models"

echo "Syncing trained model to ${HETZNER_HOST}:${REMOTE_DIR}/models/latest/ ..."
rsync -avz --delete -e "ssh ${SSH_OPTS[*]}" "$MODEL_DIR/" "${HETZNER_USER}@${HETZNER_HOST}:${REMOTE_DIR}/models/latest/"

echo "Restarting paper trading service..."
# --project-directory pins all of docker-compose.yml's relative paths
# (build context, ./models etc.) to REMOTE_DIR itself, regardless of
# where the compose file lives (docker/) - matching where this script
# just rsync'd the model to.
ssh "${SSH_OPTS[@]}" "${HETZNER_USER}@${HETZNER_HOST}" REMOTE_DIR="$REMOTE_DIR" bash <<'REMOTE_SCRIPT'
set -euo pipefail
cd "$REMOTE_DIR"
COMPOSE="docker compose -f docker/docker-compose.yml --project-directory ."
$COMPOSE up -d astra-paper astra-dashboard
$COMPOSE restart astra-paper astra-dashboard
$COMPOSE ps
REMOTE_SCRIPT

echo "Deploy complete."
