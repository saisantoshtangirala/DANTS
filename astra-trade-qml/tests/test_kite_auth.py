from unittest.mock import MagicMock

import pytest
import requests

from src.utils.kite_auth import (
    KiteLoginError,
    _extract_request_token,
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
    session.get = lambda *a, **k: MagicMock(headers={})

    with pytest.raises(KiteLoginError):
        _extract_request_token(session, "fake_api_key")


def test_extract_request_token_stops_after_max_hops():
    session = requests.Session()
    # Every hop redirects further without ever carrying a request_token.
    session.get = lambda *a, **k: MagicMock(headers={"Location": "https://kite.zerodha.com/next"})

    with pytest.raises(KiteLoginError):
        _extract_request_token(session, "fake_api_key")


def test_generate_access_token_from_env_requires_all_fields():
    with pytest.raises(KiteLoginError, match="Missing required environment variables"):
        generate_access_token_from_env(env={"KITE_API_KEY": "x"})


def test_generate_access_token_from_env_empty_raises():
    with pytest.raises(KiteLoginError):
        generate_access_token_from_env(env={})
