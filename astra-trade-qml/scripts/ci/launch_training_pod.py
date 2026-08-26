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
Reads RUNPOD_API_KEY, GH_TOKEN, and (optionally) KITE_API_KEY /
KITE_API_SECRET / KITE_USER_ID / KITE_PASSWORD / KITE_TOTP_SECRET from
the environment.
"""

import argparse
import json
import os
import re
import sys
import time

import requests

REST_BASE = "https://rest.runpod.io/v1"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def build_start_command(repo: str, branch: str, train_timeout_seconds: int) -> str:
    return f"""set -uxo pipefail
mkdir -p /workspace
cd /workspace

# CUDA diagnostics and initialization - RunPod containers sometimes need
# explicit driver init before PyTorch can see the GPU
echo "=== GPU diagnostics ==="
nvidia-smi || echo "WARNING: nvidia-smi failed"
ldconfig 2>/dev/null || true
python3 -c "import torch; torch.cuda.init(); print(f'CUDA available: {{torch.cuda.is_available()}}, device: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}')" || echo "WARNING: CUDA init check failed"

git clone --branch {branch} --single-branch "https://x-access-token:${{GH_TOKEN}}@github.com/{repo}.git" repo
cd repo/astra-trade-qml

pip install --no-cache-dir -q -r requirements/requirements-runpod-image.txt

# Verify CUDA after pip install (packages can affect torch's CUDA detection)
python3 -c "
import torch
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

timeout {train_timeout_seconds} python3 -m src.main --mode train
TRAIN_EXIT=$?

mkdir -p models/latest logs
{{
  echo "exit=$TRAIN_EXIT"
  echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}} > logs/last_run_status.txt

git config user.email "runpod-bot@astra-trade-qml.local"
git config user.name "RunPod Training Bot"
git checkout -B model-artifacts
git add -f models/latest logs
git commit -m "Automated training run $(date -u +%Y-%m-%dT%H:%M:%SZ) (exit=$TRAIN_EXIT)"
git push "https://x-access-token:${{GH_TOKEN}}@github.com/{repo}.git" HEAD:model-artifacts --force

POD_ID="${{RUNPOD_POD_ID:-$(hostname)}}"
curl -sS -X DELETE -H "Authorization: Bearer ${{RUNPOD_API_KEY}}" "https://rest.runpod.io/v1/pods/${{POD_ID}}"
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
    cloud_type: str = "COMMUNITY",
) -> dict:
    payload = {
        "name": "astra-trade-qml-training",
        "imageName": image,
        "gpuTypeIds": gpu_ids,
        "gpuCount": 1,
        "cloudType": cloud_type,
        "containerDiskInGb": container_disk_gb,
        "env": env,
        "dockerStartCmd": ["bash", "-c", start_command],
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


def wait_for_pod_boot(api_key: str, pod_id: str, boot_timeout_seconds: int, poll_interval: int = 15) -> bool:
    """
    Wait for the pod's container to actually start (RunPod populates
    `runtime` on the pod once it does - there's no other public signal,
    since RunPod's API doesn't expose pod logs). Returns False if the
    pod never boots within boot_timeout_seconds, or vanishes (404)
    before booting - both indicate a launch-time failure (e.g. the
    "layer does not exist" container-create error), not a training
    failure, and should be handled by discarding this pod and trying
    a fresh one rather than waiting out the full training timeout.
    """
    start = time.time()
    while True:
        elapsed = time.time() - start
        status = get_pod_status(api_key, pod_id)

        if status is None:
            print(f"Pod {pod_id} vanished before booting (elapsed={elapsed:.0f}s) - launch failure")
            return False

        runtime = status.get("runtime")
        desired_status = status.get("desiredStatus")
        last_status_change = status.get("lastStatusChange")
        print(
            f"Pod {pod_id} boot check: elapsed={elapsed:.0f}s desiredStatus={desired_status} "
            f"runtime={'present' if runtime else 'none'} lastStatusChange={last_status_change!r}"
        )

        if runtime:
            print(f"Pod {pod_id} booted after {elapsed:.0f}s")
            return True

        if elapsed > boot_timeout_seconds:
            print(f"Pod {pod_id} did not boot within {boot_timeout_seconds}s - treating as a launch failure")
            return False

        time.sleep(poll_interval)


def poll_until_terminated(api_key: str, pod_id: str, timeout_seconds: int, poll_interval: int = 30) -> bool:
    """Returns True if the pod self-terminated normally, False if force-stopped on timeout."""
    start = time.time()
    while True:
        elapsed = time.time() - start
        status = get_pod_status(api_key, pod_id)

        if status is None:
            print(f"Pod {pod_id} terminated after {elapsed:.0f}s")
            return True

        if elapsed > timeout_seconds:
            print(f"Timeout after {elapsed:.0f}s - force-stopping pod {pod_id}")
            terminate_pod(api_key, pod_id)
            return False

        print(
            f"Pod {pod_id} still running: elapsed={elapsed:.0f}s desiredStatus={status.get('desiredStatus')} "
            f"lastStatusChange={status.get('lastStatusChange')!r}"
        )
        time.sleep(poll_interval)


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
) -> bool:
    """
    Create a pod and wait for it to boot; if it never boots (a
    launch-time/host-side failure), discard it and retry on a fresh pod
    up to max_launch_attempts times. Once a pod boots, hand off to
    poll_until_terminated for the (much longer) training-completion wait.
    Returns True only if a pod both booted and self-terminated
    normally within its timeout.
    """
    for attempt in range(1, max_launch_attempts + 1):
        pod = create_pod(api_key, image, gpu_ids, pod_env, start_command, container_disk_gb=container_disk_gb)
        pod_id = pod["id"]
        print(
            f"Launch attempt {attempt}/{max_launch_attempts}: created pod {pod_id} "
            f"(gpu={pod.get('machine', {}).get('gpuTypeId')}, cost/hr={pod.get('costPerHr')})"
        )

        if wait_for_pod_boot(api_key, pod_id, boot_timeout_seconds=boot_timeout_seconds):
            return poll_until_terminated(api_key, pod_id, timeout_seconds=train_poll_timeout_seconds)

        print(f"Pod {pod_id} failed to boot (likely a RunPod host-side issue) - terminating and retrying")
        terminate_pod(api_key, pod_id)

    print(f"Gave up after {max_launch_attempts} launch attempts - pod never booted", file=sys.stderr)
    return False


def check_training_exit_code(repo: str, branch: str, gh_token: str) -> "int | None":
    """
    Read back logs/last_run_status.txt from the model-artifacts branch
    the pod just pushed, and return the training process's own exit
    code. Returns None if the marker is missing (the pod never got far
    enough to write it - already surfaced separately by
    launch_and_wait's boot-failure handling, so this is a defense-in-
    depth check, not the primary signal).
    """
    # The GitHub Contents API (not raw.githubusercontent.com, which doesn't
    # reliably authenticate against private repos) with the raw media type
    # returns the file body directly.
    url = f"https://api.github.com/repos/{repo}/contents/astra-trade-qml/logs/last_run_status.txt"
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {gh_token}", "Accept": "application/vnd.github.raw+json"},
            params={"ref": "model-artifacts"},
            timeout=15,
        )
    except (requests.ConnectionError, requests.Timeout) as e:
        print(f"Warning: could not fetch last_run_status.txt to verify training exit code: {e}")
        return None

    if response.status_code != 200:
        print(f"Warning: last_run_status.txt not found on model-artifacts ({response.status_code}) for branch {branch}")
        return None

    match = re.search(r"^exit=(\d+)$", response.text, re.MULTILINE)
    if not match:
        print(f"Warning: could not parse exit code from last_run_status.txt: {response.text!r}")
        return None
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="RunPod base image (default: the validated public runpod/pytorch image)")
    parser.add_argument("--repo", required=True, help="owner/repo, for the results-push clone")
    parser.add_argument("--branch", required=True, help="Branch the pod trains against")
    parser.add_argument("--gpu-ids-json", required=True, help='JSON array of GPU type IDs, e.g. select_gpu.py output')
    parser.add_argument("--container-disk-gb", type=int, default=40)
    parser.add_argument("--boot-timeout-seconds", type=int, default=600, help="Max time to wait for the container to actually start (10m default - large PyTorch images can take 5+ min to pull on first boot)")
    parser.add_argument("--max-launch-attempts", type=int, default=3, help="Retries if the pod fails to boot (host-side failures like image-layer corruption)")
    parser.add_argument("--train-timeout-seconds", type=int, default=10800, help="Hard cap inside the pod (3h default)")
    parser.add_argument("--poll-timeout-seconds", type=int, default=11400, help="Outer safety-net cap once booted (3h10m default)")
    args = parser.parse_args()

    runpod_key = os.environ.get("RUNPOD_API_KEY")
    gh_token = os.environ.get("GH_TOKEN")
    if not runpod_key or not gh_token:
        print("RUNPOD_API_KEY and GH_TOKEN environment variables are required", file=sys.stderr)
        sys.exit(1)

    gpu_ids = json.loads(args.gpu_ids_json)

    pod_env = {
        "RUNPOD_API_KEY": runpod_key,
        "GH_TOKEN": gh_token,
    }
    for kite_var in ["KITE_API_KEY", "KITE_API_SECRET", "KITE_USER_ID", "KITE_PASSWORD", "KITE_TOTP_SECRET"]:
        if os.environ.get(kite_var):
            pod_env[kite_var] = os.environ[kite_var]

    start_command = build_start_command(args.repo, args.branch, args.train_timeout_seconds)

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
    )
    if not completed:
        print("::error::Training pod never completed - either it repeatedly failed to boot or timed out mid-training", file=sys.stderr)
        sys.exit(1)

    exit_code = check_training_exit_code(args.repo, args.branch, gh_token)
    if exit_code is None:
        print("::error::Pod terminated but left no readable last_run_status.txt - treating as a failure", file=sys.stderr)
        sys.exit(1)
    if exit_code != 0:
        print(f"::error::Training process inside the pod exited with code {exit_code}", file=sys.stderr)
        sys.exit(1)

    print("Training completed successfully")


if __name__ == "__main__":
    main()
