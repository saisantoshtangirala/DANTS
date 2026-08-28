#!/usr/bin/env bash
# One-time setup for the Hetzner paper-trading host.
#
# Run this once on the Hetzner server (as root or with sudo) to create:
#   - An 'astra' system user
#   - A Python virtualenv with paper-trading dependencies
#   - Two systemd services: astra-paper (headless trader) and astra-dashboard (Streamlit)
#   - An .env file template for Kite credentials
#
# Usage (from the repo root on the Hetzner server):
#   sudo bash astra-trade-qml/scripts/hetzner_setup.sh
#
# After setup, deployments are handled by scripts/deploy_to_hetzner.sh
# (called from CI or manually) which rsyncs code + model and restarts services.
set -euo pipefail

INSTALL_DIR="/opt/astra-trade-qml"
VENV_DIR="$INSTALL_DIR/venv"
SVC_USER="astra"

echo "=== Astra-Trade QML: Hetzner one-time setup ==="

# Create service user (no login shell, no home dir collision)
if ! id "$SVC_USER" &>/dev/null; then
    useradd --system --no-create-home --shell /usr/sbin/nologin "$SVC_USER"
    echo "Created system user: $SVC_USER"
else
    echo "User $SVC_USER already exists"
fi

# Create directory structure
mkdir -p "$INSTALL_DIR"/{src,config,models/latest,data,logs,requirements,scripts}
chown -R "$SVC_USER:$SVC_USER" "$INSTALL_DIR"

# Install Python 3.10+ and venv if missing
if ! command -v python3 &>/dev/null; then
    apt-get update && apt-get install -y python3 python3-venv python3-pip
fi

# Create virtualenv
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    echo "Created virtualenv at $VENV_DIR"
else
    echo "Virtualenv already exists at $VENV_DIR"
fi

# Install dependencies if requirements file exists
if [[ -f "$INSTALL_DIR/requirements/requirements-paper.txt" ]]; then
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r "$INSTALL_DIR/requirements/requirements-paper.txt"
    echo "Installed Python dependencies"
fi

# Create .env template (don't overwrite if it already has real values)
ENV_FILE="$INSTALL_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    cat > "$ENV_FILE" <<'ENVEOF'
# Zerodha Kite credentials — fill these in once.
KITE_API_KEY=
KITE_API_SECRET=
KITE_USER_ID=
KITE_PASSWORD=
KITE_TOTP_SECRET=
ENVEOF
    chmod 600 "$ENV_FILE"
    chown "$SVC_USER:$SVC_USER" "$ENV_FILE"
    echo "Created $ENV_FILE — edit it to add your Kite credentials"
else
    echo "$ENV_FILE already exists, not overwriting"
fi

# --- systemd unit: astra-paper (headless paper trading) ---
cat > /etc/systemd/system/astra-paper.service <<UNITEOF
[Unit]
Description=Astra-Trade QML Paper Trading
After=network.target

[Service]
Type=simple
User=$SVC_USER
Group=$SVC_USER
WorkingDirectory=$INSTALL_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$VENV_DIR/bin/python3 -m src.main --mode paper
Restart=on-failure
RestartSec=30
StandardOutput=append:$INSTALL_DIR/logs/paper.log
StandardError=append:$INSTALL_DIR/logs/paper.log

[Install]
WantedBy=multi-user.target
UNITEOF

# --- systemd unit: astra-dashboard (Streamlit) ---
cat > /etc/systemd/system/astra-dashboard.service <<UNITEOF
[Unit]
Description=Astra-Trade QML Dashboard (Streamlit)
After=network.target

[Service]
Type=simple
User=$SVC_USER
Group=$SVC_USER
WorkingDirectory=$INSTALL_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$VENV_DIR/bin/streamlit run src/dashboard/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
Restart=on-failure
RestartSec=10
StandardOutput=append:$INSTALL_DIR/logs/dashboard.log
StandardError=append:$INSTALL_DIR/logs/dashboard.log

[Install]
WantedBy=multi-user.target
UNITEOF

systemctl daemon-reload
systemctl enable astra-paper astra-dashboard
echo "Systemd services created and enabled (astra-paper, astra-dashboard)"

chown -R "$SVC_USER:$SVC_USER" "$INSTALL_DIR"

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Edit $ENV_FILE with your Kite credentials"
echo "  2. Run the deploy script (or trigger CI) to sync code + model"
echo "  3. Services will start automatically after first deploy"
