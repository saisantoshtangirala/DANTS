from unittest.mock import MagicMock, patch

import pytest
import requests

from src.utils.kite_auth import (
    KiteLoginError,
    _extract_request_token,
    _totp_with_headroom,
    generate_access_token_from_env,
)


def test_extract_request_token_walks_redirect_chain():
    session = requests.Session()
    responses = [
        MagicMock(headers={"Location": "https://kite.zerodha.com/connect/finish?api_key=x&sess_id=y"}),
        MagicMock(headers={
            "Location": "https://myapp.example.com/callback?request_token=abc123def&action=login&status=success"
        }),
    ]
    calls = iter(responses)
    session.get = lambda *a, **k: next(calls)

    token = _extract_request_token(session, "fake_api_key")
    assert token == "abc123def"


def test_extract_request_token_raises_when_not_found():
    session = requests.Session()
    session.get = lambda *a, **k: MagicMock(headers={}, text="<html>some page</html>")

    with pytest.raises(KiteLoginError, match="Could not find request_token"):
        _extract_request_token(session, "fake_api_key")


def test_extract_request_token_stops_after_max_hops():
    session = requests.Session()
    session.get = lambda *a, **k: MagicMock(headers={"Location": "https://kite.zerodha.com/next"})

    with pytest.raises(KiteLoginError):
        _extract_request_token(session, "fake_api_key")


def test_extract_request_token_detects_consent_screen():
    session = requests.Session()
    responses = [
        MagicMock(headers={"Location": "https://kite.zerodha.com/connect/finish?sess_id=y"}),
        MagicMock(headers={"Location": "https://kite.zerodha.com/connect/authorize?sess_id=y"}),
        MagicMock(headers={}, text="<html>JS consent page</html>"),
    ]
    calls = iter(responses)
    session.get = lambda *a, **k: next(calls)

    with pytest.raises(KiteLoginError, match="not been authorised"):
        _extract_request_token(session, "fake_api_key")


def test_extract_request_token_finds_token_in_page_body():
    session = requests.Session()
    responses = [
        MagicMock(headers={"Location": "https://kite.zerodha.com/connect/finish?sess_id=y"}),
        MagicMock(
            headers={},
            text='<html><script>window.location="http://127.0.0.1/?request_token=xyz789&status=success"</script></html>',
        ),
    ]
    calls = iter(responses)
    session.get = lambda *a, **k: next(calls)

    token = _extract_request_token(session, "fake_api_key")
    assert token == "xyz789"


def test_totp_with_headroom_returns_code_immediately_when_enough_time():
    with patch("src.utils.kite_auth.time.time", return_value=1000.0), \
         patch("src.utils.kite_auth.time.sleep") as mock_sleep:
        mock_totp = MagicMock()
        mock_totp.interval = 30
        mock_totp.now.return_value = "123456"
        with patch("src.utils.kite_auth.pyotp.TOTP", return_value=mock_totp):
            code = _totp_with_headroom("JBSWY3DPEHPK3PXP", min_seconds=5.0)
    assert code == "123456"
    mock_sleep.assert_not_called()


def test_totp_with_headroom_waits_when_code_about_to_expire():
    # 1048.0 % 30 = 28, so remaining = 30 - 28 = 2s (< 5s headroom)
    with patch("src.utils.kite_auth.time.time", return_value=1048.0), \
         patch("src.utils.kite_auth.time.sleep") as mock_sleep:
        mock_totp = MagicMock()
        mock_totp.interval = 30
        mock_totp.now.return_value = "654321"
        with patch("src.utils.kite_auth.pyotp.TOTP", return_value=mock_totp):
            code = _totp_with_headroom("JBSWY3DPEHPK3PXP", min_seconds=5.0)
    assert code == "654321"
    mock_sleep.assert_called_once()
    wait_arg = mock_sleep.call_args[0][0]
    assert wait_arg > 2.0


def test_generate_access_token_from_env_requires_all_fields():
    with pytest.raises(KiteLoginError, match="Missing required environment variables"):
        generate_access_token_from_env(env={"KITE_API_KEY": "x"})


def test_generate_access_token_from_env_empty_raises():
    with pytest.raises(KiteLoginError):
        generate_access_token_from_env(env={})
