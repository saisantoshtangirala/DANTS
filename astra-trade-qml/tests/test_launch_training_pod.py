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
    assert "python3 -u -m src.main --mode train" in cmd


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


def test_build_start_command_runs_cuda_diagnostics_before_training():
    cmd = ltp.build_start_command("owner/repo", "main", 10800)
    assert "nvidia-smi" in cmd
    assert "ldconfig" in cmd
    train_pos = cmd.index("python3 -u -m src.main --mode train")
    nvidia_pos = cmd.index("nvidia-smi")
    assert nvidia_pos < train_pos


def test_build_start_command_pushes_to_model_artifacts_additively_not_blind_force():
    """
    Regression test: two pods (e.g. a production training run and a
    swing-test diagnostic run) each start from their own fresh clone
    with no idea what the other has already pushed to model-artifacts.
    A blind `git checkout -B model-artifacts && git push --force` means
    whichever pod finishes last silently wipes out the other's pushed
    artifacts. The pod must instead fetch the branch, build its commit
    on top of whatever's already there, and push non-force - falling
    back to --force only as a last resort after repeated non-force
    push rejections.
    """
    cmd = ltp.build_start_command("owner/repo", "main", 10800)

    fetch_pos = cmd.index("git fetch")
    non_force_push_pos = cmd.index('git push "$REPO_URL" HEAD:model-artifacts')
    assert fetch_pos < non_force_push_pos, "must fetch model-artifacts before pushing to it"

    # The primary push path must not pass --force.
    non_force_push_line = cmd.splitlines()[cmd[:non_force_push_pos].count("\n")]
    assert "--force" not in non_force_push_line

    # A --force push must still exist, but only in the last-resort
    # fallback branch, after the retry loop and its warning.
    fallback_pos = cmd.index('git push "$REPO_URL" HEAD:model-artifacts --force')
    warning_pos = cmd.index("falling back to a force push")
    assert warning_pos < fallback_pos
    assert non_force_push_pos < fallback_pos

    # New artifacts must be staged before any branch switch, so a
    # `git checkout` onto model-artifacts's already-committed files at
    # the same paths can't collide with our freshly-written ones.
    stage_pos = cmd.index("ARTIFACT_STAGE=$(mktemp -d)")
    checkout_pos = cmd.index("git checkout -B model-artifacts")
    assert stage_pos < checkout_pos


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


def test_terminate_stale_pods_by_name_terminates_matching_pods():
    """
    Regression test: a pod that leaks past a cancelled run's cleanup
    (SIGKILL racing the SIGINT handler's DELETE call) sits on RunPod's
    infrastructure indefinitely under the fixed pod name, burning
    GPU-hours - a fresh launch must always sweep for one first.
    """
    pods_response = MagicMock(
        status_code=200,
        json=lambda: [
            {"id": "stale1", "name": "astra-trade-qml-training"},
            {"id": "other", "name": "some-other-pod"},
            {"id": "stale2", "name": "astra-trade-qml-training"},
        ],
    )
    with patch("launch_training_pod._request_with_retries", return_value=pods_response), \
         patch("launch_training_pod.terminate_pod") as mock_terminate:
        ltp.terminate_stale_pods_by_name("key", "astra-trade-qml-training")

    terminated_ids = {call.args[1] for call in mock_terminate.call_args_list}
    assert terminated_ids == {"stale1", "stale2"}


def test_terminate_stale_pods_by_name_noop_when_none_match():
    pods_response = MagicMock(status_code=200, json=lambda: [{"id": "other", "name": "some-other-pod"}])
    with patch("launch_training_pod._request_with_retries", return_value=pods_response), \
         patch("launch_training_pod.terminate_pod") as mock_terminate:
        ltp.terminate_stale_pods_by_name("key", "astra-trade-qml-training")

    mock_terminate.assert_not_called()


def test_terminate_stale_pods_by_name_handles_dict_response_shape():
    """RunPod's list-pods response shape (bare list vs {"pods": [...]}) isn't
    guaranteed - handle both rather than assuming one."""
    pods_response = MagicMock(
        status_code=200,
        json=lambda: {"pods": [{"id": "stale1", "name": "astra-trade-qml-training"}]},
    )
    with patch("launch_training_pod._request_with_retries", return_value=pods_response), \
         patch("launch_training_pod.terminate_pod") as mock_terminate:
        ltp.terminate_stale_pods_by_name("key", "astra-trade-qml-training")

    mock_terminate.assert_called_once_with("key", "stale1")


def test_terminate_stale_pods_by_name_survives_list_failure():
    """A failure to list pods must not crash the launch - it's a
    best-effort sweep, not a hard prerequisite."""
    with patch("launch_training_pod._request_with_retries", side_effect=RuntimeError("boom")), \
         patch("launch_training_pod.terminate_pod") as mock_terminate:
        ltp.terminate_stale_pods_by_name("key", "astra-trade-qml-training")  # must not raise

    mock_terminate.assert_not_called()


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


def test_wait_for_pod_boot_returns_true_if_pod_vanishes_late():
    """Pod ran so fast it vanished before we saw runtime, but elapsed > 90s means it likely completed."""
    with patch("launch_training_pod.get_pod_status", return_value=None), \
         patch("launch_training_pod.time.sleep"), \
         patch("launch_training_pod.time.time", side_effect=[0, 100]):
        result = ltp.wait_for_pod_boot("key", "pod123", boot_timeout_seconds=1000, poll_interval=0)

    assert result is True


def test_wait_for_pod_boot_returns_true_on_exited_status():
    """Pod already finished its start command — EXITED means it booted and ran."""
    with patch("launch_training_pod.get_pod_status") as mock_status, patch("launch_training_pod.time.sleep"):
        mock_status.side_effect = [
            {"desiredStatus": "RUNNING", "runtime": None},
            {"desiredStatus": "EXITED", "runtime": None},
        ]
        result = ltp.wait_for_pod_boot("key", "pod123", boot_timeout_seconds=1000, poll_interval=0)

    assert result is True


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
    mock_poll.assert_called_once_with("key", "pod2", timeout_seconds=1000, repo="", gh_token="")


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


def test_check_fresh_completion_marker_detects_marker_after_since():
    from datetime import datetime, timezone
    since = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with patch("launch_training_pod.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200, text="exit=0\nfinished_at=2026-01-01T00:30:00Z\n",
        )
        result = ltp.check_fresh_completion_marker("owner/repo", "token", since)

    assert result == datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)


def test_check_fresh_completion_marker_ignores_stale_marker():
    """A marker left over from a previous run (finished before this pod
    even launched) must not be mistaken for the current pod completing."""
    from datetime import datetime, timezone
    since = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)  # pod launched at 1am
    with patch("launch_training_pod.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200, text="exit=0\nfinished_at=2026-01-01T00:30:00Z\n",  # marker from before launch
        )
        result = ltp.check_fresh_completion_marker("owner/repo", "token", since)

    assert result is None


def test_check_fresh_completion_marker_returns_none_when_marker_missing():
    with patch("launch_training_pod.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=404, text="Not Found")
        from datetime import datetime, timezone
        result = ltp.check_fresh_completion_marker("owner/repo", "token", datetime.now(timezone.utc))

    assert result is None


def test_poll_until_terminated_force_terminates_on_fresh_git_marker():
    """
    Regression test for the real failure this was built for: a pod's
    self-DELETE call can succeed from RunPod's API perspective while the
    underlying pod resource never tears down, leaving desiredStatus
    stuck at RUNNING forever even though training finished and pushed
    its results. The git marker must be treated as authoritative and
    force-terminate the pod rather than waiting on status forever.
    """
    from datetime import datetime, timezone
    launched_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with patch("launch_training_pod.get_pod_status", return_value={"desiredStatus": "RUNNING"}), \
         patch("launch_training_pod.check_fresh_completion_marker",
               return_value=datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)), \
         patch("launch_training_pod.terminate_pod") as mock_terminate, \
         patch("launch_training_pod.time.sleep"), \
         patch("launch_training_pod.time.time", side_effect=[0, 61]):
        result = ltp.poll_until_terminated(
            "key", "pod123", timeout_seconds=1000, poll_interval=0,
            repo="owner/repo", gh_token="token", launched_at=launched_at,
        )

    assert result is True
    mock_terminate.assert_called_once_with("key", "pod123")


def test_poll_until_terminated_ignores_git_marker_when_repo_not_given():
    """Without repo/gh_token, the git-marker fallback must not be consulted
    at all (e.g. call sites that don't have GH_TOKEN available)."""
    with patch("launch_training_pod.get_pod_status") as mock_status, \
         patch("launch_training_pod.check_fresh_completion_marker") as mock_marker, \
         patch("launch_training_pod.time.sleep"):
        mock_status.side_effect = [{"desiredStatus": "RUNNING"}, None]
        result = ltp.poll_until_terminated("key", "pod123", timeout_seconds=1000, poll_interval=0)

    assert result is True
    mock_marker.assert_not_called()
