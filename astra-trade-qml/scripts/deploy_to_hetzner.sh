#!/usr/bin/env bash
# Deploys the latest trained model to the Hetzner paper-trading host and
# restarts the services so they pick it up.
#
# Expects to be run from a checkout of the `model-artifacts` branch (i.e.
# astra-trade-qml/models/latest/ present in the working directory - that's
# what the RunPod training pod pushes there; see scripts/ci/launch_training_pod.py).
#
# Uses systemd services (astra-paper, astra-dashboard) running in a Python
# virtualenv — no Docker required. Run scripts/hetzner_setup.sh once on the
# host before the first deploy.
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
SSH_CMD="ssh ${SSH_OPTS[*]}"

echo "Ensuring remote directories exist..."
ssh "${SSH_OPTS[@]}" "${HETZNER_USER}@${HETZNER_HOST}" \
  "mkdir -p ${REMOTE_DIR}/{models/latest,src,config,requirements,scripts,logs,data}"

echo "Syncing trained model..."
rsync -avz --delete -e "$SSH_CMD" \
  "$MODEL_DIR/" "${HETZNER_USER}@${HETZNER_HOST}:${REMOTE_DIR}/models/latest/"

echo "Syncing source code and config..."
rsync -avz --delete -e "$SSH_CMD" \
  "astra-trade-qml/src/" "${HETZNER_USER}@${HETZNER_HOST}:${REMOTE_DIR}/src/"
rsync -avz --delete -e "$SSH_CMD" \
  "astra-trade-qml/config/" "${HETZNER_USER}@${HETZNER_HOST}:${REMOTE_DIR}/config/"
rsync -avz -e "$SSH_CMD" \
  "astra-trade-qml/requirements/requirements-paper.txt" \
  "${HETZNER_USER}@${HETZNER_HOST}:${REMOTE_DIR}/requirements/"

echo "Installing/updating Python dependencies..."
ssh "${SSH_OPTS[@]}" "${HETZNER_USER}@${HETZNER_HOST}" bash <<REMOTE_DEPS
set -euo pipefail
cd "$REMOTE_DIR"
if [[ ! -d venv ]]; then
    python3.11 -m venv venv
    echo "Created virtualenv"
fi
venv/bin/pip install --quiet --upgrade pip
venv/bin/pip install --quiet -r requirements/requirements-paper.txt
echo "Dependencies up to date"
REMOTE_DEPS

echo "Restarting services..."
ssh "${SSH_OPTS[@]}" "${HETZNER_USER}@${HETZNER_HOST}" bash <<'REMOTE_SVC'
set -euo pipefail
sudo systemctl restart astra-paper astra-dashboard
sleep 2
echo "Service status:"
sudo systemctl is-active astra-paper astra-dashboard || true
REMOTE_SVC

echo "Deploy complete."
