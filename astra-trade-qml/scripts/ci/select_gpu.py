"""
Pick the cheapest currently-in-stock RunPod GPU meeting this project's
minimum VRAM floor, via RunPod's GraphQL pricing/stock API.

The quantum layer (quantum_kernel.py / vqc_classifier.py) runs
AerSimulator's statevector method on CPU regardless of GPU - only the
LSTM benefits from GPU acceleration - so the VRAM floor here is modest
(12GB) rather than tuned for the quantum workload.

Usage: RUNPOD_API_KEY=... python3 select_gpu.py [--min-vram-gb 12] [--limit 5]
Prints a JSON array of GPU type IDs, cheapest first, to stdout - this is
the priority-ordered list RunPod's pod-creation API expects for
`gpuTypeIds` (it tries each in order until one is actually available).
"""

import argparse
import json
import os
import sys

import requests

_GRAPHQL_URL = "https://api.runpod.io/graphql"
_QUERY = """
query GpuTypes {
  gpuTypes {
    id
    displayName
    memoryInGb
    communityCloud
    secureCloud
    lowestPrice(input: { gpuCount: 1 }) {
      uninterruptablePrice
      stockStatus
    }
  }
}
"""


def select_gpus(api_key: str, min_vram_gb: int = 12, limit: int = 5) -> list:
    response = requests.post(
        _GRAPHQL_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"query": _QUERY},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    if "errors" in data:
        raise RuntimeError(f"RunPod GraphQL error: {data['errors']}")

    gpus = data["data"]["gpuTypes"]
    candidates = []

    for gpu in gpus:
        lowest_price = gpu.get("lowestPrice") or {}
        price = lowest_price.get("uninterruptablePrice")
        stock = lowest_price.get("stockStatus")

        if gpu["memoryInGb"] < min_vram_gb:
            continue
        if price is None or stock in (None, "None"):
            continue

        candidates.append((price, gpu["id"]))

    candidates.sort()
    return [gpu_id for _, gpu_id in candidates[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-vram-gb", type=int, default=12)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("RUNPOD_API_KEY environment variable is required", file=sys.stderr)
        sys.exit(1)

    gpu_ids = select_gpus(api_key, min_vram_gb=args.min_vram_gb, limit=args.limit)
    if not gpu_ids:
        print(f"No GPUs found with >= {args.min_vram_gb}GB VRAM currently in stock", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(gpu_ids))


if __name__ == "__main__":
    main()
