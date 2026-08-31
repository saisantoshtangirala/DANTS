import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ci"))

import select_gpu  # noqa: E402


FAKE_GPU_TYPES_RESPONSE = {
    "data": {
        "gpuTypes": [
            {
                "id": "RTX 3080 Ti", "memoryInGb": 12, "communityCloud": True, "secureCloud": False,
                "lowestPrice": {"uninterruptablePrice": 0.18, "stockStatus": "Low"},
            },
            {
                "id": "RTX 4090", "memoryInGb": 24, "communityCloud": True, "secureCloud": True,
                "lowestPrice": {"uninterruptablePrice": 0.34, "stockStatus": "Low"},
            },
            {
                "id": "RTX 3070", "memoryInGb": 8, "communityCloud": True, "secureCloud": False,
                "lowestPrice": {"uninterruptablePrice": 0.10, "stockStatus": "Low"},
            },
            {
                "id": "H100", "memoryInGb": 80, "communityCloud": True, "secureCloud": True,
                "lowestPrice": {"uninterruptablePrice": None, "stockStatus": None},
            },
        ]
    }
}


def test_select_gpus_filters_and_sorts():
    with patch("select_gpu.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: FAKE_GPU_TYPES_RESPONSE, raise_for_status=lambda: None
        )
        result = select_gpu.select_gpus("fake-key", min_vram_gb=12, limit=5)

    assert result == ["RTX 3080 Ti", "RTX 4090"]


def test_select_gpus_respects_limit():
    with patch("select_gpu.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: FAKE_GPU_TYPES_RESPONSE, raise_for_status=lambda: None
        )
        result = select_gpu.select_gpus("fake-key", min_vram_gb=12, limit=1)

    assert result == ["RTX 3080 Ti"]


def test_select_gpus_raises_on_graphql_error():
    with patch("select_gpu.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {"errors": [{"message": "bad key"}]}, raise_for_status=lambda: None
        )
        try:
            select_gpu.select_gpus("bad-key")
            assert False, "should have raised"
        except RuntimeError as e:
            assert "bad key" in str(e)


def test_select_gpus_empty_when_nothing_meets_floor():
    with patch("select_gpu.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: FAKE_GPU_TYPES_RESPONSE, raise_for_status=lambda: None
        )
        result = select_gpu.select_gpus("fake-key", min_vram_gb=200, limit=5)

    assert result == []
