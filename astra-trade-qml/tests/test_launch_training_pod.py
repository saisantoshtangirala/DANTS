import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ci"))

import launch_training_pod as ltp  # noqa: E402


def test_build_start_command_includes_repo_branch_and_timeout():
    cmd = ltp.build_start_command("owner/repo", "main", 10800)
    assert "owner/repo" in cmd
    assert "--branch main" in cmd
    assert "timeout 10800" in cmd
    assert "model-artifacts" in cmd
    assert "RUNPOD_POD_ID" in cmd  # self-terminate step present


def test_build_start_command_clones_and_installs_deps_no_custom_image():
    """No CI-built Docker image involved - the pod itself clones and pip installs."""
    cmd = ltp.build_start_command("owner/repo", "main", 10800)
    assert "git clone --branch main --single-branch" in cmd
    assert "pip install" in cmd
    assert "requirements-runpod-image.txt" in cmd
    assert "python3 -m src.main --mode train" in cmd


def test_build_start_command_always_writes_last_run_status():
    """
    Regression test: the pod must always leave a marker on model-artifacts,
    even if training crashed immediately - otherwise a failed run that got
    far enough to boot but not far enough to train leaves no trace at all
    (git commit no-ops when nothing changed).
    """
    cmd = ltp.build_start_command("owner/repo", "main", 10800)
    assert "logs/last_run_status.txt" in cmd
    assert "exit=$TRAIN_EXIT" in cmd


def test_default_image_is_public_runpod_base():
    assert ltp.DEFAULT_IMAGE == "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def test_request_with_retries_retries_on_connection_error_then_succeeds():
    ok_response = MagicMock(status_code=200)
    with patch("launch_training_pod.requests.request", side_effect=[requests.ConnectionError("boom"), ok_response]), \
         patch("launch_training_pod.time.sleep"):
        result = ltp._request_with_retries("GET", "https://example.com")

    assert result is ok_response


def test_request_with_retries_retries_on_5xx_then_succeeds():
    bad_response = MagicMock(status_code=502, text="bad gateway")
    ok_response = MagicMock(status_code=200)
    with patch("launch_training_pod.requests.request", side_effect=[bad_response, ok_response]), \
         patch("launch_training_pod.time.sleep"):
        result = ltp._request_with_retries("GET", "https://example.com")

    assert result is ok_response


def test_request_with_retries_does_not_retry_4xx():
    bad_response = MagicMock(status_code=404, text="not found")
    with patch("launch_training_pod.requests.request", return_value=bad_response) as mock_request, \
         patch("launch_training_pod.time.sleep"):
        result = ltp._request_with_retries("GET", "https://example.com")

    assert result is bad_response
    assert mock_request.call_count == 1


def test_request_with_retries_raises_after_exhausting_attempts():
    with patch("launch_training_pod.requests.request", side_effect=requests.ConnectionError("boom")), \
         patch("launch_training_pod.time.sleep"):
        try:
            ltp._request_with_retries("GET", "https://example.com", max_attempts=3)
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "3 attempts" in str(e)


def test_create_pod_returns_pod_metadata():
    with patch("launch_training_pod.requests.request") as mock_request:
        mock_request.return_value = MagicMock(status_code=200, json=lambda: {"id": "pod123", "costPerHr": 0.18})
        result = ltp.create_pod("key", "image:tag", ["gpu1"], {"X": "y"}, "echo hi")

    assert result["id"] == "pod123"
    payload = mock_request.call_args.kwargs["json"]
    assert payload["imageName"] == "image:tag"
    assert payload["gpuTypeIds"] == ["gpu1"]
    assert payload["dockerStartCmd"] == ["bash", "-c", "echo hi"]


def test_create_pod_raises_with_error_body_on_failure():
    with patch("launch_training_pod.requests.request") as mock_request:
        mock_request.return_value = MagicMock(status_code=400, text="no instances available for gpuTypeIds")
        try:
            ltp.create_pod("key", "image:tag", ["gpu1"], {}, "echo hi")
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "no instances available" in str(e)


def test_get_pod_status_returns_none_on_404():
    with patch("launch_training_pod.requests.request") as mock_request:
        mock_request.return_value = MagicMock(status_code=404)
        assert ltp.get_pod_status("key", "pod123") is None


def test_get_pod_status_returns_json_body():
    with patch("launch_training_pod.requests.request") as mock_request:
        mock_request.return_value = MagicMock(status_code=200, json=lambda: {"desiredStatus": "RUNNING"})
        assert ltp.get_pod_status("key", "pod123") == {"desiredStatus": "RUNNING"}


def test_wait_for_pod_boot_returns_true_once_runtime_present():
    with patch("launch_training_pod.get_pod_status") as mock_status, patch("launch_training_pod.time.sleep"):
        mock_status.side_effect = [
            {"desiredStatus": "RUNNING", "runtime": None},
            {"desiredStatus": "RUNNING", "runtime": {"uptimeInSeconds": 5}},
        ]
        result = ltp.wait_for_pod_boot("key", "pod123", boot_timeout_seconds=1000, poll_interval=0)

    assert result is True


def test_wait_for_pod_boot_returns_false_on_timeout():
    with patch("launch_training_pod.get_pod_status") as mock_status, \
         patch("launch_training_pod.time.sleep"), \
         patch("launch_training_pod.time.time", side_effect=[0, 0, 5, 20]):
        mock_status.return_value = {"desiredStatus": "RUNNING", "runtime": None}
        result = ltp.wait_for_pod_boot("key", "pod123", boot_timeout_seconds=10, poll_interval=0)

    assert result is False


def test_wait_for_pod_boot_returns_false_if_pod_vanishes():
    """
    Regression test for the real failure this was built for: a container
    that fails to create (e.g. "layer does not exist" on a corrupted
    RunPod host) can leave the pod object gone before it ever boots.
    """
    with patch("launch_training_pod.get_pod_status", return_value=None), patch("launch_training_pod.time.sleep"):
        result = ltp.wait_for_pod_boot("key", "pod123", boot_timeout_seconds=1000, poll_interval=0)

    assert result is False


def test_poll_until_terminated_returns_true_on_termination():
    with patch("launch_training_pod.get_pod_status") as mock_status, patch("launch_training_pod.time.sleep"):
        mock_status.side_effect = [{"desiredStatus": "RUNNING"}, None]
        result = ltp.poll_until_terminated("key", "pod123", timeout_seconds=1000, poll_interval=0)

    assert result is True


def test_poll_until_terminated_force_stops_on_timeout():
    with patch("launch_training_pod.get_pod_status") as mock_status, \
         patch("launch_training_pod.terminate_pod") as mock_terminate, \
         patch("launch_training_pod.time.sleep"), \
         patch("launch_training_pod.time.time", side_effect=[0, 0, 5, 20]):
        mock_status.return_value = {"desiredStatus": "RUNNING"}
        result = ltp.poll_until_terminated("key", "pod123", timeout_seconds=10, poll_interval=0)

    assert result is False
    assert mock_terminate.called


def test_launch_and_wait_retries_after_boot_failure_then_succeeds():
    """
    The scenario this whole module was rebuilt for: the first pod fails
    to boot (host-side container-create error), gets discarded, and a
    second pod boots and completes fine.
    """
    with patch("launch_training_pod.create_pod") as mock_create, \
         patch("launch_training_pod.wait_for_pod_boot", side_effect=[False, True]), \
         patch("launch_training_pod.terminate_pod") as mock_terminate, \
         patch("launch_training_pod.poll_until_terminated", return_value=True) as mock_poll:
        mock_create.side_effect = [{"id": "pod1"}, {"id": "pod2"}]

        result = ltp.launch_and_wait(
            "key", "image", ["gpu1"], {}, "cmd", container_disk_gb=40,
            boot_timeout_seconds=60, train_poll_timeout_seconds=1000, max_launch_attempts=3,
        )

    assert result is True
    assert mock_create.call_count == 2
    mock_terminate.assert_called_once_with("key", "pod1")
    mock_poll.assert_called_once_with("key", "pod2", timeout_seconds=1000)


def test_launch_and_wait_gives_up_after_max_attempts():
    with patch("launch_training_pod.create_pod", return_value={"id": "pod1"}), \
         patch("launch_training_pod.wait_for_pod_boot", return_value=False), \
         patch("launch_training_pod.terminate_pod"), \
         patch("launch_training_pod.poll_until_terminated") as mock_poll:
        result = ltp.launch_and_wait(
            "key", "image", ["gpu1"], {}, "cmd", container_disk_gb=40,
            boot_timeout_seconds=60, train_poll_timeout_seconds=1000, max_launch_attempts=3,
        )

    assert result is False
    assert not mock_poll.called


def test_check_training_exit_code_parses_marker_file():
    with patch("launch_training_pod.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, text="exit=0\nfinished_at=2026-01-01T00:00:00Z\n")
        assert ltp.check_training_exit_code("owner/repo", "main", "token") == 0


def test_check_training_exit_code_detects_nonzero_exit():
    with patch("launch_training_pod.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200, text="exit=1\nfinished_at=2026-01-01T00:00:00Z\n")
        assert ltp.check_training_exit_code("owner/repo", "main", "token") == 1


def test_check_training_exit_code_returns_none_when_marker_missing():
    with patch("launch_training_pod.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=404, text="Not Found")
        assert ltp.check_training_exit_code("owner/repo", "main", "token") is None
