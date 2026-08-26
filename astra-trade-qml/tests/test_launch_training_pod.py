import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ci"))

import launch_training_pod as ltp  # noqa: E402


def test_build_start_command_includes_repo_branch_and_timeout():
    cmd = ltp.build_start_command("owner/repo", "main", 10800)
    assert "owner/repo" in cmd
    assert "--branch main" in cmd
    assert "timeout 10800" in cmd
    assert "model-artifacts" in cmd
    assert "RUNPOD_POD_ID" in cmd  # self-terminate step present


def test_create_pod_returns_pod_metadata():
    with patch("launch_training_pod.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            json=lambda: {"id": "pod123", "costPerHr": 0.18}, raise_for_status=lambda: None
        )
        result = ltp.create_pod("key", "image:tag", ["gpu1"], {"X": "y"}, "echo hi")

    assert result["id"] == "pod123"
    payload = mock_post.call_args.kwargs["json"]
    assert payload["imageName"] == "image:tag"
    assert payload["gpuTypeIds"] == ["gpu1"]
    assert payload["dockerStartCmd"] == ["bash", "-c", "echo hi"]


def test_poll_until_terminated_returns_true_on_404():
    with patch("launch_training_pod.requests.get") as mock_get, patch("launch_training_pod.time.sleep"):
        mock_get.side_effect = [MagicMock(status_code=200), MagicMock(status_code=404)]
        result = ltp.poll_until_terminated("key", "pod123", timeout_seconds=1000, poll_interval=0)

    assert result is True


def test_poll_until_terminated_force_stops_on_timeout():
    with patch("launch_training_pod.requests.get") as mock_get, \
         patch("launch_training_pod.requests.delete") as mock_delete, \
         patch("launch_training_pod.time.sleep"), \
         patch("launch_training_pod.time.time", side_effect=[0, 0, 5, 20]):
        mock_get.return_value = MagicMock(status_code=200)
        result = ltp.poll_until_terminated("key", "pod123", timeout_seconds=10, poll_interval=0)

    assert result is False
    assert mock_delete.called
