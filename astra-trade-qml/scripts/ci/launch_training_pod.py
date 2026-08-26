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

Usage:
    python3 launch_training_pod.py \\
        --repo owner/repo --branch main \\
        --gpu-ids-json '["RTX 3080 Ti","RTX 4090"]' \\
        --timeout-seconds 10800
Reads RUNPOD_API_KEY, GH_TOKEN, and (optionally) KITE_API_KEY /
KITE_API_SECRET / KITE_USER_ID / KITE_PASSWORD / KITE_TOTP_SECRET from
the environment.
"""

import argparse
import json
import os
import sys
import time

import requests

REST_BASE = "https://rest.runpod.io/v1"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def build_start_command(repo: str, branch: str, train_timeout_seconds: int) -> str:
    return f"""set -uxo pipefail
mkdir -p /workspace
cd /workspace
git clone --branch {branch} --single-branch "https://x-access-token:${{GH_TOKEN}}@github.com/{repo}.git" repo
cd repo/astra-trade-qml

pip install --no-cache-dir -q -r requirements/requirements-runpod-image.txt

timeout {train_timeout_seconds} python3 -m src.main --mode train
TRAIN_EXIT=$?

git config user.email "runpod-bot@astra-trade-qml.local"
git config user.name "RunPod Training Bot"
git checkout -B model-artifacts
git add -f models/latest logs
git commit -m "Automated training run $(date -u +%Y-%m-%dT%H:%M:%SZ) (exit=$TRAIN_EXIT)" || echo "nothing to commit"
git push "https://x-access-token:${{GH_TOKEN}}@github.com/{repo}.git" HEAD:model-artifacts --force

POD_ID="${{RUNPOD_POD_ID:-$(hostname)}}"
curl -sS -X DELETE -H "Authorization: Bearer ${{RUNPOD_API_KEY}}" "https://rest.runpod.io/v1/pods/${{POD_ID}}"
"""


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
    response = requests.post(
        f"{REST_BASE}/pods",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def poll_until_terminated(api_key: str, pod_id: str, timeout_seconds: int, poll_interval: int = 30) -> bool:
    """Returns True if the pod self-terminated normally, False if force-stopped on timeout."""
    start = time.time()
    while True:
        elapsed = time.time() - start
        response = requests.get(
            f"{REST_BASE}/pods/{pod_id}", headers={"Authorization": f"Bearer {api_key}"}, timeout=15
        )

        if response.status_code == 404:
            print(f"Pod {pod_id} terminated after {elapsed:.0f}s")
            return True

        if elapsed > timeout_seconds:
            print(f"Timeout after {elapsed:.0f}s - force-stopping pod {pod_id}")
            requests.delete(
                f"{REST_BASE}/pods/{pod_id}", headers={"Authorization": f"Bearer {api_key}"}, timeout=15
            )
            return False

        print(f"Pod {pod_id} still running, elapsed={elapsed:.0f}s")
        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="RunPod base image (default: the validated public runpod/pytorch image)")
    parser.add_argument("--repo", required=True, help="owner/repo, for the results-push clone")
    parser.add_argument("--branch", required=True, help="Branch the pod trains against")
    parser.add_argument("--gpu-ids-json", required=True, help='JSON array of GPU type IDs, e.g. select_gpu.py output')
    parser.add_argument("--container-disk-gb", type=int, default=40)
    parser.add_argument("--train-timeout-seconds", type=int, default=10800, help="Hard cap inside the pod (3h default)")
    parser.add_argument("--poll-timeout-seconds", type=int, default=11400, help="Outer safety-net cap (3h10m default)")
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

    pod = create_pod(
        runpod_key,
        args.image,
        gpu_ids,
        pod_env,
        start_command,
        container_disk_gb=args.container_disk_gb,
    )
    pod_id = pod["id"]
    print(f"Created pod {pod_id} (gpu={pod.get('machine', {}).get('gpuTypeId')}, cost/hr={pod.get('costPerHr')})")

    success = poll_until_terminated(runpod_key, pod_id, timeout_seconds=args.poll_timeout_seconds)
    if not success:
        print("Training pod did not complete within the timeout and was force-stopped", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
