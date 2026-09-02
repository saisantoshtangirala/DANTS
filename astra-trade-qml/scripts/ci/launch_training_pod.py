"""
Launch a RunPod training pod, wait for it to self-terminate (success or
failure), and exit non-zero if it doesn't complete within the
safety-net timeout (in which case the pod is force-stopped here so it
can't keep billing).

No custom Docker image or registry involved, deliberately: this is the
exact pattern already validated end-to-end on real GPU hardware for the
one-off smoke test (scripts/runpod_smoke_test.py) - a public RunPod
base image, with the pod itself cloning the repo and installing
dependencies at startup. A CI-built custom image was tried first for a
faster/more reproducible pod boot, but broke twice in ways that took
longer to chase than the few minutes a fresh clone + pip install costs
per run, so this reverts to the proven approach.

Robustness notes (learned from a real failure: "error creating
container: container: create: container create: Error response from
daemon: layer does not exist" - a corrupted Docker image-layer cache on
whichever RunPod worker host the pod happened to land on, unrelated to
anything in this repo):

- RunPod's own API has no pod-logs endpoint (confirmed unavailable as
  of writing - see runpod/runpod-python#400), so a container that fails
  to start is otherwise invisible to CI beyond "the pod exists but never
  runs our start command." wait_for_pod_boot() below treats the
  appearance of `runtime` in the pod's status as the boot signal, with a
  short, separate timeout from the long training-completion wait - a
  pod that never boots is retried on a fresh pod (almost always a
  different worker host) rather than silently burning the full
  poll-timeout-seconds budget waiting for a container that will never
  start.
- All RunPod API calls retry with backoff on transient HTTP/network
  errors (5xx, timeouts, connection errors) - a blip in RunPod's API
  used to kill the whole run immediately via an unhandled exception.
- The pod always leaves a logs/last_run_status.txt marker on
  model-artifacts, even if training crashed immediately - previously,
  a pod that failed before writing anything meaningful left
  model-artifacts unchanged with no trace of having run at all (`git
  commit` no-ops when nothing changed). main() reads this marker back
  after the pod terminates and fails CI if the training process itself
  exited non-zero, instead of only checking "did the pod eventually
  disappear."

Usage:
    python3 launch_training_pod.py \\
        --repo owner/repo --branch main \\
        --gpu-ids-json '["RTX 3080 Ti","RTX 4090"]' \\
        --train-timeout-seconds 10800
Reads RUNPOD_API_KEY, GH_TOKEN, and (optionally) RUNPOD_SSH_KEY,
KITE_API_KEY / KITE_API_SECRET / KITE_USER_ID / KITE_PASSWORD /
KITE_TOTP_SECRET from the environment.
"""

import argparse
import atexit
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone

import requests

REST_BASE = "https://rest.runpod.io/v1"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

# Tracks the pod that should be terminated if this process is killed
# (e.g. GitHub Actions sends SIGINT on job cancellation, then SIGKILL
# after a 7.5s grace period). Both the atexit hook and the signal
# handlers check this.
_active_pod: dict | None = None  # {"api_key": ..., "pod_id": ...}


def _cleanup_active_pod() -> None:
    """Terminate the active RunPod pod if one exists. Safe to call multiple times."""
    global _active_pod
    pod = _active_pod
    if pod is None:
        return
    _active_pod = None
    print(f"Cleanup: terminating active pod {pod['pod_id']}", flush=True)
    try:
        requests.delete(
            f"{REST_BASE}/pods/{pod['pod_id']}",
            headers={"Authorization": f"Bearer {pod['api_key']}"},
            timeout=5,
        )
    except Exception as e:
        print(f"Cleanup DELETE failed: {e}", flush=True)


def _cancel_handler(signum, frame) -> None:
    print(f"Received signal {signum} - cleaning up RunPod pod before exit", flush=True)
    _cleanup_active_pod()
    sys.exit(128 + signum)


atexit.register(_cleanup_active_pod)
signal.signal(signal.SIGTERM, _cancel_handler)
signal.signal(signal.SIGINT, _cancel_handler)


def build_start_command(repo: str, branch: str, train_timeout_seconds: int, script_mode: str = "train") -> str:
    # "train" keeps the original, unprefixed paths - check_training_exit_code()/
    # check_fresh_completion_marker() hardcode logs/last_run_status.txt for the
    # production training path. Any other script_mode (e.g. swing-test) gets its
    # own prefixed paths so a run in that mode never collides with (or is
    # mistaken for) a concurrent production training run's status/log on the
    # shared model-artifacts branch - the exact class of bug
    # check_fresh_completion_marker's since-timestamp guard exists to prevent,
    # here avoided by not sharing a path at all.
    status_file = "logs/last_run_status.txt" if script_mode == "train" else f"logs/{script_mode}_last_run_status.txt"
    log_file_dest = "logs/training_full.log" if script_mode == "train" else f"logs/{script_mode}_training_full.log"
    return f"""set -uxo pipefail

# Start sshd for CI log streaming — dockerStartCmd overrides the
# container's default CMD so RunPod's init script that normally starts
# sshd doesn't run.
if [ -n "${{SSH_PUBLIC_KEY:-}}" ]; then
    mkdir -p ~/.ssh /var/run/sshd
    echo "$SSH_PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/authorized_keys
    ssh-keygen -A 2>&1 || true
    if [ -f /etc/ssh/sshd_config ]; then
        sed -i 's/^#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
        sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
    fi
    /usr/sbin/sshd && echo "sshd started for log streaming" || echo "WARNING: sshd failed to start"
fi

mkdir -p /workspace
cd /workspace

# Create the log file immediately so SSH log streamer starts showing
# output from boot (not just from when training starts).
LOG_FILE=/workspace/training.log
touch "$LOG_FILE"

# From here, tee all stdout+stderr to the log file so the SSH
# streamer sees git clone, pip install, GPU diagnostics, and any
# crash messages — not just training output.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== GPU diagnostics (pre-install) ==="
nvidia-smi || echo "WARNING: nvidia-smi failed"
ldconfig 2>/dev/null || true
# Do NOT call torch.cuda.init() here — pip install below may change
# CUDA libraries, corrupting an already-initialized CUDA context and
# causing "CUDA unknown error" for the entire rest of the process.

set +x  # hide token from trace
git clone --branch {branch} --single-branch "https://x-access-token:${{GH_TOKEN}}@github.com/{repo}.git" repo
set -x
cd repo/astra-trade-qml

pip install --no-cache-dir -q -r requirements/requirements-runpod-image.txt

# First CUDA init happens here, AFTER pip install — safe to initialize.
echo "=== GPU diagnostics (post-install) ==="
python3 -c "
import torch
torch.cuda.init()
print(f'PyTorch {{torch.__version__}}, CUDA available: {{torch.cuda.is_available()}}')
if torch.cuda.is_available():
    print(f'GPU: {{torch.cuda.get_device_name(0)}}, VRAM: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}} GB')
else:
    print('WARNING: CUDA not available - all training will run on CPU')
try:
    import xgboost as xgb
    print(f'XGBoost {{xgb.__version__}}')
except: pass
"

git config user.email "runpod-bot@astra-trade-qml.local"
git config user.name "RunPod Training Bot"

set +x  # hide token from trace
REPO_URL="https://x-access-token:${{GH_TOKEN}}@github.com/{repo}.git"
set -x

echo "=== Starting training ==="
PYTHONUNBUFFERED=1 timeout {train_timeout_seconds} python3 -u -m src.main --mode {script_mode} 2>&1
TRAIN_EXIT=$?

mkdir -p models/latest logs
{{
  echo "exit=$TRAIN_EXIT"
  echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}} > {status_file}
cp "$LOG_FILE" {log_file_dest} 2>/dev/null || true

# Stage our new output files outside the working tree before touching
# branches: model-artifacts already has committed files at these same
# paths (models/latest/*, logs/*) from earlier runs, and a plain
# `git checkout` onto it would abort ("untracked working tree files
# would be overwritten") with our freshly-written, not-yet-committed
# files sitting right there.
ARTIFACT_STAGE=$(mktemp -d)
mkdir -p "$ARTIFACT_STAGE/models/latest" "$ARTIFACT_STAGE/logs"
cp -a models/latest/. "$ARTIFACT_STAGE/models/latest/" 2>/dev/null || true
cp -a logs/. "$ARTIFACT_STAGE/logs/" 2>/dev/null || true

# Fetch-and-merge instead of a blind force-push: two pods (e.g. a
# production training run and a swing-test diagnostic run, or two
# production runs queued back-to-back) each start from their own fresh
# clone with no idea what the other has already pushed. A blind
# `git checkout -B model-artifacts && git push --force` here means
# whichever pod finishes last silently wipes out the other's pushed
# models/logs. Each script_mode writes to its own paths (train:
# models/latest/, logs/last_run_status.txt, logs/training_full.log;
# swing-test: logs/swing-test_*), so there's never a real content
# conflict to resolve — layering our new files on top of whatever is
# already on the branch and pushing non-force is enough. Retry with
# backoff on a rejected push (another pod won the race and moved the
# branch tip since our fetch); only fall back to --force if every
# retry is rejected, so a run still completes rather than hanging
# forever on a pathological, repeatedly-losing race.
# (REPO_URL was already set above, before training started.)

PUSH_OK=0
for attempt in 1 2 3 4 5; do
  set +x  # hide token from trace
  if git fetch "$REPO_URL" model-artifacts 2>/dev/null; then
    set -x
    git checkout -B model-artifacts FETCH_HEAD
  else
    set -x
    echo "No existing model-artifacts branch found remotely (or fetch failed) - creating it fresh"
    git checkout -B model-artifacts
  fi

  mkdir -p models/latest logs
  cp -a "$ARTIFACT_STAGE/models/latest/." models/latest/
  cp -a "$ARTIFACT_STAGE/logs/." logs/
  git add -f models/latest logs
  git commit --allow-empty -m "Automated training run $(date -u +%Y-%m-%dT%H:%M:%SZ) (exit=$TRAIN_EXIT)"

  set +x  # hide token from trace
  if git push "$REPO_URL" HEAD:model-artifacts; then
    PUSH_OK=1
    set -x
    break
  fi
  set -x
  echo "Push to model-artifacts rejected on attempt $attempt/5 (likely a concurrent pod pushed first) - retrying..."
  sleep $((5 * attempt))
done

if [ "$PUSH_OK" != "1" ]; then
  echo "WARNING: non-force push to model-artifacts failed after 5 attempts - falling back to a force push (may overwrite a concurrent pod's changes)"
  set +x  # hide token from trace
  git push "$REPO_URL" HEAD:model-artifacts --force
  set -x
fi

POD_ID="${{RUNPOD_POD_ID:-$(hostname)}}"
echo "Training done (exit=$TRAIN_EXIT). Self-terminating pod $POD_ID..."
curl -sS -X DELETE -H "Authorization: Bearer ${{RUNPOD_API_KEY}}" "https://rest.runpod.io/v1/pods/${{POD_ID}}" || true

# Prevent RunPod from restarting the container after the command exits.
# The DELETE above is async — the pod may still be running when it returns.
# Without this, the container exits cleanly and RunPod's restart policy
# re-executes dockerStartCmd, causing a duplicate training run.
echo "Waiting for pod termination..."
sleep infinity
"""


def _request_with_retries(method: str, url: str, max_attempts: int = 4, backoff_seconds: float = 2.0, **kwargs):
    """
    Retry transient failures (connection errors, timeouts, 5xx) with
    exponential backoff. Does NOT retry 4xx responses - those are real
    errors (bad payload, auth, 404) that won't fix themselves.
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.request(method, url, **kwargs)
        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = e
        else:
            if response.status_code < 500:
                return response
            last_error = RuntimeError(f"{response.status_code} from {url}: {response.text[:500]}")

        if attempt < max_attempts:
            sleep_for = backoff_seconds * (2 ** (attempt - 1))
            print(f"Request to {url} failed ({last_error}), retrying in {sleep_for:.0f}s (attempt {attempt}/{max_attempts})")
            time.sleep(sleep_for)

    raise RuntimeError(f"Request to {url} failed after {max_attempts} attempts: {last_error}")


def create_pod(
    api_key: str,
    image: str,
    gpu_ids: list,
    env: dict,
    start_command: str,
    container_disk_gb: int = 40,
    cloud_type: str = "SECURE",
    pod_name: str = "astra-trade-qml-training",
) -> dict:
    payload = {
        "name": pod_name,
        "imageName": image,
        "gpuTypeIds": gpu_ids,
        "gpuCount": 1,
        "cloudType": cloud_type,
        "containerDiskInGb": container_disk_gb,
        "env": env,
        "dockerStartCmd": ["bash", "-c", start_command],
        "ports": ["22/tcp"],
    }
    response = _request_with_retries(
        "POST",
        f"{REST_BASE}/pods",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"RunPod pod creation failed ({response.status_code}): {response.text[:1000]}")
    return response.json()


def terminate_pod(api_key: str, pod_id: str) -> None:
    try:
        _request_with_retries(
            "DELETE", f"{REST_BASE}/pods/{pod_id}", headers={"Authorization": f"Bearer {api_key}"}, timeout=15
        )
    except RuntimeError as e:
        # Best-effort - if this fails the pod may already be gone, or
        # RunPod is having its own issues; either way there's nothing
        # more useful to do here than log it and move on.
        print(f"Warning: failed to terminate pod {pod_id}: {e}")


def terminate_stale_pods_by_name(api_key: str, name: str) -> None:
    """
    Terminate any existing pod with this fixed name before creating a new
    one. Cleanup on cancellation (_cancel_handler / atexit) is
    best-effort: GitHub Actions sends SIGINT then SIGKILL after a ~7.5s
    grace period, and if that DELETE call doesn't finish in time (or the
    signal doesn't propagate before SIGKILL), the pod itself keeps
    running on RunPod's infrastructure - it isn't tied to the CI
    runner's lifetime once launched. Since every pod uses the same fixed
    name, a leaked one from a prior cancelled run would otherwise sit
    there burning GPU-hours indefinitely. This makes a fresh run
    self-healing regardless of why a previous cleanup didn't run: always
    start by clearing anything already using this name.
    """
    try:
        response = _request_with_retries(
            "GET", f"{REST_BASE}/pods", headers={"Authorization": f"Bearer {api_key}"}, timeout=15
        )
    except RuntimeError as e:
        print(f"Warning: could not list existing pods to check for stale ones: {e}")
        return

    if response.status_code >= 400:
        print(f"Warning: listing pods failed ({response.status_code}): {response.text[:300]}")
        return

    pods = response.json()
    if isinstance(pods, dict):
        pods = pods.get("pods", pods.get("data", []))

    stale = [p for p in pods if p.get("name") == name]
    for pod in stale:
        pod_id = pod.get("id")
        print(f"Found stale pod {pod_id} named {name!r} from a previous run - terminating before launch", flush=True)
        terminate_pod(api_key, pod_id)


def get_pod_status(api_key: str, pod_id: str):
    """Returns the pod's status dict, or None if it's already gone (404)."""
    response = _request_with_retries(
        "GET", f"{REST_BASE}/pods/{pod_id}", headers={"Authorization": f"Bearer {api_key}"}, timeout=15
    )
    if response.status_code == 404:
        return None
    if response.status_code >= 400:
        raise RuntimeError(f"RunPod get-pod failed ({response.status_code}): {response.text[:500]}")
    return response.json()


def _ssh_probe(host: str, port: int, ssh_key_path: str) -> bool:
    """Try a quick SSH connection to see if the pod's sshd is reachable."""
    if not ssh_key_path or not host:
        return False
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                "-o", "ConnectTimeout=10",
                "-i", ssh_key_path,
                "-p", str(port),
                f"root@{host}",
                "echo ok",
            ],
            capture_output=True, timeout=15,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            print(f"SSH probe stderr: {stderr}", flush=True)
        return result.returncode == 0
    except Exception as e:
        print(f"SSH probe exception: {e}", flush=True)
        return False


def _parse_ssh_port(port_mappings) -> int | None:
    """Extract the host-side SSH port from RunPod's portMappings field."""
    if not port_mappings:
        return None
    if isinstance(port_mappings, list):
        for pm in port_mappings:
            if isinstance(pm, dict):
                if pm.get("containerPort") == 22 or pm.get("privatePort") == 22:
                    return pm.get("hostPort") or pm.get("publicPort")
            elif isinstance(pm, str) and "22/tcp" in pm:
                match = re.search(r":(\d+)$", pm)
                if match:
                    return int(match.group(1))
    elif isinstance(port_mappings, dict):
        if "22" in port_mappings:
            return port_mappings["22"]
        if "22/tcp" in port_mappings:
            return port_mappings["22/tcp"]
    return None


def wait_for_pod_boot(
    api_key: str, pod_id: str, boot_timeout_seconds: int, poll_interval: int = 15,
    ssh_key_path: str = "",
) -> dict | None:
    """
    Wait for the pod's container to start and become SSH-reachable.

    Polls the REST API for publicIp + portMappings (exposed by the
    ``"ports": ["22/tcp"]`` flag in the create payload). The port
    mapping typically appears long before the container actually
    starts (while the image is still pulling), so the SSH probe runs
    on every poll cycle until sshd responds or the boot timeout
    expires.

    Returns ``{"host": ip, "port": mapped_port}`` on success, or
    None on boot failure.
    """
    start = time.time()
    _logged_api_response = False
    _logged_port_mappings = False
    ssh_host = ""
    ssh_port = 0
    while True:
        elapsed = time.time() - start
        status = get_pod_status(api_key, pod_id)

        if status is None:
            if elapsed > 90:
                print(f"Pod {pod_id} vanished after {elapsed:.0f}s — likely completed its start command")
                return {"host": ssh_host, "port": ssh_port}
            print(f"Pod {pod_id} vanished before booting (elapsed={elapsed:.0f}s) - launch failure")
            return None

        desired_status = status.get("desiredStatus")
        public_ip = status.get("publicIp")
        port_mappings = status.get("portMappings")

        if not _logged_api_response:
            _logged_api_response = True
            print(f"Pod {pod_id} API response keys: {sorted(status.keys())}", flush=True)

        if port_mappings and not _logged_port_mappings:
            _logged_port_mappings = True
            print(f"Pod {pod_id} portMappings (raw): {port_mappings!r}", flush=True)

        if public_ip and port_mappings and not ssh_host:
            parsed_port = _parse_ssh_port(port_mappings)
            if parsed_port:
                ssh_host = public_ip
                ssh_port = int(parsed_port)
                print(f"Pod {pod_id}: SSH endpoint discovered: {ssh_host}:{ssh_port}", flush=True)

        print(
            f"Pod {pod_id} boot check: elapsed={elapsed:.0f}s desiredStatus={desired_status} "
            f"publicIp={public_ip!r} sshPort={ssh_port or None!r}"
        , flush=True)

        if desired_status == "EXITED":
            print(f"Pod {pod_id} already exited after {elapsed:.0f}s — it booted and completed")
            return {"host": ssh_host, "port": ssh_port}

        if ssh_host and ssh_port and ssh_key_path:
            if _ssh_probe(ssh_host, ssh_port, ssh_key_path):
                print(f"Pod {pod_id} booted and SSH-reachable after {elapsed:.0f}s")
                return {"host": ssh_host, "port": ssh_port}

        if elapsed > boot_timeout_seconds:
            if ssh_host:
                print(
                    f"Pod {pod_id}: boot timeout after {elapsed:.0f}s — sshd never responded at "
                    f"{ssh_host}:{ssh_port}, returning endpoint anyway for streamer retries"
                )
                return {"host": ssh_host, "port": ssh_port}
            print(f"Pod {pod_id} did not boot within {boot_timeout_seconds}s - treating as a launch failure")
            return None

        time.sleep(poll_interval)


class SSHLogStreamer:
    """Stream a pod's training log to stdout via SSH in a background thread."""

    def __init__(self, host: str, port: int, ssh_key_path: str, log_path: str = "/workspace/training.log"):
        self._host = host
        self._port = port
        self._ssh_key_path = ssh_key_path
        self._log_path = log_path
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        tail_cmd = f"while [ ! -f {self._log_path} ]; do sleep 5; done; tail -n +1 -f {self._log_path}"
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-o", "ConnectTimeout=30",
            "-i", self._ssh_key_path,
            "-p", str(self._port),
            f"root@{self._host}",
            tail_cmd,
        ]

        max_retries = 20
        for attempt in range(1, max_retries + 1):
            if self._stop.is_set():
                return
            try:
                print(f"SSH log stream: connecting to root@{self._host}:{self._port} (attempt {attempt}/{max_retries})...", flush=True)
                self._proc = subprocess.Popen(
                    ssh_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                print("SSH log stream: connected — streaming pod output ↓", flush=True)
                for line in iter(self._proc.stdout.readline, b""):
                    if self._stop.is_set():
                        return
                    sys.stdout.buffer.write(line)
                    sys.stdout.buffer.flush()
                self._proc.wait()
                if self._stop.is_set():
                    return
                stderr = self._proc.stderr.read().decode(errors="replace").strip()
                if stderr:
                    print(f"SSH log stream: connection closed ({stderr})", flush=True)
                else:
                    print("SSH log stream: connection closed", flush=True)
            except Exception as e:
                print(f"SSH log stream: error ({e})", flush=True)

            if attempt < max_retries and not self._stop.is_set():
                wait = min(30, 10 * attempt)
                print(f"SSH log stream: retrying in {wait}s...", flush=True)
                self._stop.wait(wait)


def poll_until_terminated(
    api_key: str, pod_id: str, timeout_seconds: int, poll_interval: int = 30,
    ssh_key_path: str = "", ssh_host: str = "", ssh_port: int = 0,
    repo: str = "", gh_token: str = "", launched_at: "datetime | None" = None,
    status_file: str = "logs/last_run_status.txt",
) -> bool:
    """Returns True if the pod finished (self-terminated, or we detected
    completion via git and force-terminated it ourselves), False if
    force-stopped on timeout without ever finishing."""
    streamer = None
    if ssh_key_path and ssh_host and ssh_port:
        streamer = SSHLogStreamer(ssh_host, ssh_port, ssh_key_path)
        streamer.start()

    check_git_marker = bool(repo and gh_token and launched_at is not None)
    marker_check_interval = 60  # git-marker polling doesn't need every 30s cycle
    last_marker_check = 0.0

    try:
        start = time.time()
        while True:
            elapsed = time.time() - start

            # Primary signal: the pod's own self-DELETE / status change.
            status = get_pod_status(api_key, pod_id)

            if status is None:
                print(f"Pod {pod_id} terminated after {elapsed:.0f}s")
                return True

            desired = status.get("desiredStatus")
            if desired == "EXITED":
                print(f"Pod {pod_id} exited after {elapsed:.0f}s — cleaning up")
                terminate_pod(api_key, pod_id)
                return True

            # Secondary, independent signal: the pod already pushed its
            # results and a completion marker to model-artifacts. A
            # pod's self-DELETE call is "curl ... || true" - it can
            # succeed from RunPod's API perspective while the underlying
            # pod resource never actually tears down, which would
            # otherwise strand this loop waiting forever on a pod that
            # has nothing left to do. Git has pushed reliably every time
            # this session, so trust it and terminate the pod ourselves
            # rather than wait on a self-termination that may never land.
            if check_git_marker and elapsed - last_marker_check >= marker_check_interval:
                last_marker_check = elapsed
                finished_at = check_fresh_completion_marker(repo, gh_token, launched_at, status_file=status_file)
                if finished_at is not None:
                    print(
                        f"Pod {pod_id}: training finished at {finished_at.isoformat()} per "
                        f"model-artifacts (self-termination hasn't taken effect after {elapsed:.0f}s) "
                        f"— force-terminating now",
                        flush=True,
                    )
                    terminate_pod(api_key, pod_id)
                    return True

            if elapsed > timeout_seconds:
                print(f"Timeout after {elapsed:.0f}s - force-stopping pod {pod_id}")
                terminate_pod(api_key, pod_id)
                return False

            if not streamer:
                print(
                    f"Pod {pod_id} still running: elapsed={elapsed:.0f}s desiredStatus={desired} "
                    f"lastStatusChange={status.get('lastStatusChange')!r}"
                )

            time.sleep(poll_interval)
    finally:
        if streamer:
            streamer.stop()


def launch_and_wait(
    api_key: str,
    image: str,
    gpu_ids: list,
    pod_env: dict,
    start_command: str,
    container_disk_gb: int,
    boot_timeout_seconds: int,
    train_poll_timeout_seconds: int,
    max_launch_attempts: int = 3,
    cloud_type: str = "SECURE",
    fallback_cloud_type: str = "COMMUNITY",
    ssh_key_path: str = "",
    repo: str = "",
    gh_token: str = "",
    pod_name: str = "astra-trade-qml-training",
    status_file: str = "logs/last_run_status.txt",
) -> bool:
    """
    Create a pod and wait for it to boot; if it never boots (a
    launch-time/host-side failure), discard it and retry on a fresh pod
    up to max_launch_attempts times. Once a pod boots, hand off to
    poll_until_terminated for the (much longer) training-completion wait.
    Returns True only if a pod both booted and completed (self-terminated,
    or was force-terminated once we detected it had finished via git)
    within its timeout.

    repo/gh_token, when given, let poll_until_terminated cross-check
    completion against the model-artifacts branch instead of relying
    solely on the pod's own (unreliable - see check_fresh_completion_marker)
    self-termination.

    If every attempt on cloud_type exhausts without a pod ever booting
    (observed in practice as RunPod's SECURE cloud returning "no
    instances currently available" on every retry - a capacity issue,
    not something max_launch_attempts alone can work around since it's
    still the same limited pool of hosts), the whole attempt loop is
    retried once more on fallback_cloud_type before giving up entirely.
    Pass fallback_cloud_type="" (or equal to cloud_type) to disable this.
    """
    global _active_pod

    clouds_to_try = [cloud_type]
    if fallback_cloud_type and fallback_cloud_type != cloud_type:
        clouds_to_try.append(fallback_cloud_type)

    for cloud in clouds_to_try:
        for attempt in range(1, max_launch_attempts + 1):
            try:
                pod = create_pod(api_key, image, gpu_ids, pod_env, start_command, container_disk_gb=container_disk_gb, cloud_type=cloud, pod_name=pod_name)
            except RuntimeError as e:
                print(f"Launch attempt {attempt}/{max_launch_attempts} on {cloud} cloud: create_pod failed ({e})")
                continue

            pod_id = pod["id"]
            _active_pod = {"api_key": api_key, "pod_id": pod_id}
            launched_at = datetime.now(timezone.utc)
            print(
                f"Launch attempt {attempt}/{max_launch_attempts}: created pod {pod_id} on {cloud} cloud "
                f"(gpu={pod.get('machine', {}).get('gpuTypeId')}, cost/hr={pod.get('costPerHr')})"
            )

            boot_info = wait_for_pod_boot(api_key, pod_id, boot_timeout_seconds=boot_timeout_seconds, ssh_key_path=ssh_key_path)
            if boot_info is not None:
                result = poll_until_terminated(
                    api_key, pod_id, timeout_seconds=train_poll_timeout_seconds,
                    ssh_key_path=ssh_key_path,
                    ssh_host=boot_info.get("host", ""),
                    ssh_port=boot_info.get("port", 0),
                    repo=repo, gh_token=gh_token, launched_at=launched_at,
                    status_file=status_file,
                )
                _active_pod = None
                return result

            print(f"Pod {pod_id} failed to boot (likely a RunPod host-side issue) - terminating and retrying")
            terminate_pod(api_key, pod_id)
            _active_pod = None

        if cloud != clouds_to_try[-1]:
            print(f"Gave up after {max_launch_attempts} launch attempts on {cloud} cloud - trying {clouds_to_try[clouds_to_try.index(cloud) + 1]} cloud next")

    print(f"Gave up after {max_launch_attempts} launch attempts per cloud ({', '.join(clouds_to_try)}) - pod never booted", file=sys.stderr)
    return False


def _fetch_last_run_status_text(repo: str, gh_token: str, status_file: str = "logs/last_run_status.txt") -> "str | None":
    """Fetch a status file's raw content (default: logs/last_run_status.txt,
    the production training path - pass a different status_file for a
    non-"train" script_mode, e.g. swing-test's own prefixed path - see
    build_start_command) from the model-artifacts branch via the GitHub
    Contents API, or None if it's missing/unreachable.

    The Contents API (not raw.githubusercontent.com, which doesn't
    reliably authenticate against private repos) with the raw media type
    returns the file body directly.
    """
    url = f"https://api.github.com/repos/{repo}/contents/astra-trade-qml/{status_file}"
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {gh_token}", "Accept": "application/vnd.github.raw+json"},
            params={"ref": "model-artifacts"},
            timeout=15,
        )
    except (requests.ConnectionError, requests.Timeout) as e:
        print(f"Warning: could not fetch last_run_status.txt: {e}")
        return None

    if response.status_code != 200:
        return None
    return response.text


def check_training_exit_code(
    repo: str, branch: str, gh_token: str, status_file: str = "logs/last_run_status.txt"
) -> "int | None":
    """
    Read back the status file (default logs/last_run_status.txt) from
    the model-artifacts branch the pod just pushed, and return the
    training process's own exit code. Returns None if the marker is
    missing (the pod never got far enough to write it - already
    surfaced separately by launch_and_wait's boot-failure handling, so
    this is a defense-in-depth check, not the primary signal).
    """
    text = _fetch_last_run_status_text(repo, gh_token, status_file=status_file)
    if text is None:
        print(f"Warning: {status_file} not found on model-artifacts for branch {branch}")
        return None

    match = re.search(r"^exit=(\d+)$", text, re.MULTILINE)
    if not match:
        print(f"Warning: could not parse exit code from last_run_status.txt: {text!r}")
        return None
    return int(match.group(1))


def check_fresh_completion_marker(
    repo: str, gh_token: str, since: datetime, status_file: str = "logs/last_run_status.txt"
) -> "datetime | None":
    """
    Check whether the status file (default logs/last_run_status.txt) on
    model-artifacts shows a finished_at timestamp after `since` - i.e.
    the CURRENT pod pushed a completion marker, not a stale one left
    over from an earlier run. Returns the parsed finished_at datetime if
    fresh, else None.

    This exists because a pod's self-DELETE call (curl ... || true,
    silently swallowing any failure) is not a reliable completion
    signal on its own: observed in practice, training can finish, push
    its results to model-artifacts, and call DELETE on itself
    successfully from RunPod's API perspective, while the pod's
    underlying resource never actually tears down - leaving
    poll_until_terminated's status-based wait stuck indefinitely even
    though the work is done. Git (which has pushed reliably on every
    run this session) is a more trustworthy completion signal than
    RunPod's pod-status API, so we poll it independently and force-
    terminate the pod ourselves the moment we see it, rather than
    depending on the pod's self-termination succeeding.
    """
    text = _fetch_last_run_status_text(repo, gh_token, status_file=status_file)
    if text is None:
        return None

    match = re.search(r"^finished_at=(\S+)$", text, re.MULTILINE)
    if not match:
        return None

    try:
        finished_at = datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None

    return finished_at if finished_at > since else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="RunPod base image (default: the validated public runpod/pytorch image)")
    parser.add_argument("--repo", required=True, help="owner/repo, for the results-push clone")
    parser.add_argument("--branch", required=True, help="Branch the pod trains against")
    parser.add_argument("--gpu-ids-json", required=True, help='JSON array of GPU type IDs, e.g. select_gpu.py output')
    parser.add_argument("--container-disk-gb", type=int, default=40)
    parser.add_argument("--cloud-type", default="SECURE", choices=["SECURE", "COMMUNITY"], help="RunPod cloud type (SECURE = RunPod datacenters with pre-cached images, COMMUNITY = third-party hosts)")
    parser.add_argument("--fallback-cloud-type", default="COMMUNITY", choices=["SECURE", "COMMUNITY", ""], help="If every launch attempt on --cloud-type exhausts (e.g. RunPod capacity errors), retry the same attempt budget on this cloud before giving up. Pass '' to disable.")
    parser.add_argument("--boot-timeout-seconds", type=int, default=1800, help="Max time to wait for the container to start (30m default - the ~15GB PyTorch image can take 15-25min to pull on cold hosts)")
    parser.add_argument("--max-launch-attempts", type=int, default=3, help="Retries if the pod fails to boot (host-side failures like image-layer corruption)")
    parser.add_argument("--train-timeout-seconds", type=int, default=10800, help="Hard cap inside the pod (3h default)")
    parser.add_argument("--script-mode", default="train", choices=["train", "swing-test"], help="src.main --mode to run inside the pod (default: train)")
    parser.add_argument("--pod-name", default="", help="RunPod pod name (default: astra-trade-qml-training for train mode, astra-trade-qml-{script-mode} otherwise, so a non-train run never collides with a concurrent production training pod)")
    parser.add_argument("--poll-timeout-seconds", type=int, default=11400, help="Outer safety-net cap once booted (3h10m default)")
    parser.add_argument("--ssh-key-file", default="", help="Path to SSH private key for log streaming (overrides RUNPOD_SSH_KEY env var)")
    args = parser.parse_args()

    runpod_key = os.environ.get("RUNPOD_API_KEY")
    gh_token = os.environ.get("GH_TOKEN")
    if not runpod_key or not gh_token:
        print("RUNPOD_API_KEY and GH_TOKEN environment variables are required", file=sys.stderr)
        sys.exit(1)

    ssh_key_path = args.ssh_key_file
    ssh_key_tmpfile = None
    if not ssh_key_path:
        ssh_key_b64 = os.environ.get("RUNPOD_SSH_KEY", "").strip()
        if ssh_key_b64:
            import base64
            try:
                ssh_key_content = base64.b64decode(ssh_key_b64).decode()
            except Exception:
                ssh_key_content = ssh_key_b64
            ssh_key_tmpfile = tempfile.NamedTemporaryFile(mode="w", suffix="_runpod_ssh", delete=False)
            ssh_key_tmpfile.write(ssh_key_content if ssh_key_content.endswith("\n") else ssh_key_content + "\n")
            ssh_key_tmpfile.close()
            os.chmod(ssh_key_tmpfile.name, 0o600)
            ssh_key_path = ssh_key_tmpfile.name
            print(f"SSH log streaming: key loaded from RUNPOD_SSH_KEY env var ({ssh_key_path})")

    # Derive public key to inject into pod via SSH_PUBLIC_KEY env var
    ssh_public_key = ""
    if ssh_key_path:
        pub_path = ssh_key_path + ".pub"
        if os.path.exists(pub_path):
            with open(pub_path) as f:
                ssh_public_key = f.read().strip()
            print(f"SSH log streaming: read public key from {pub_path}")
        else:
            try:
                result = subprocess.run(
                    ["ssh-keygen", "-y", "-f", ssh_key_path],
                    capture_output=True, timeout=5,
                )
                if result.returncode == 0:
                    ssh_public_key = result.stdout.decode().strip()
                    print(f"SSH log streaming: derived public key from private key")
                else:
                    print(f"SSH log streaming: could not derive public key: {result.stderr.decode().strip()}")
            except Exception as e:
                print(f"SSH log streaming: could not derive public key: {e}")

    if not ssh_key_path:
        print("Note: no SSH key provided — pod logs will not be streamed to this terminal. "
              "Set RUNPOD_SSH_KEY or pass --ssh-key-file to enable.")

    gpu_ids = json.loads(args.gpu_ids_json)

    pod_env = {
        "RUNPOD_API_KEY": runpod_key,
        "GH_TOKEN": gh_token,
    }
    if ssh_public_key:
        pod_env["SSH_PUBLIC_KEY"] = ssh_public_key
    for kite_var in ["KITE_API_KEY", "KITE_API_SECRET", "KITE_USER_ID", "KITE_PASSWORD", "KITE_TOTP_SECRET"]:
        if os.environ.get(kite_var):
            pod_env[kite_var] = os.environ[kite_var]

    start_command = build_start_command(args.repo, args.branch, args.train_timeout_seconds, script_mode=args.script_mode)

    pod_name = args.pod_name or (
        "astra-trade-qml-training" if args.script_mode == "train" else f"astra-trade-qml-{args.script_mode}"
    )
    status_file = (
        "logs/last_run_status.txt" if args.script_mode == "train" else f"logs/{args.script_mode}_last_run_status.txt"
    )

    terminate_stale_pods_by_name(runpod_key, pod_name)

    try:
        completed = launch_and_wait(
            runpod_key,
            args.image,
            gpu_ids,
            pod_env,
            start_command,
            container_disk_gb=args.container_disk_gb,
            boot_timeout_seconds=args.boot_timeout_seconds,
            train_poll_timeout_seconds=args.poll_timeout_seconds,
            max_launch_attempts=args.max_launch_attempts,
            cloud_type=args.cloud_type,
            fallback_cloud_type=args.fallback_cloud_type,
            ssh_key_path=ssh_key_path,
            repo=args.repo,
            gh_token=gh_token,
            pod_name=pod_name,
            status_file=status_file,
        )
    finally:
        if ssh_key_tmpfile:
            try:
                os.unlink(ssh_key_tmpfile.name)
            except OSError:
                pass
    if not completed:
        print("::error::Training pod never completed - either it repeatedly failed to boot or timed out mid-training", file=sys.stderr)
        sys.exit(1)

    exit_code = check_training_exit_code(args.repo, args.branch, gh_token, status_file=status_file)
    if exit_code is None:
        print(f"::error::Pod terminated but left no readable {status_file} - treating as a failure", file=sys.stderr)
        sys.exit(1)
    if exit_code != 0:
        print(f"::error::Training process inside the pod exited with code {exit_code}", file=sys.stderr)
        sys.exit(1)

    print("Training completed successfully")


if __name__ == "__main__":
    main()
