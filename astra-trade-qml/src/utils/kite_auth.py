"""
Automated Zerodha Kite Connect login.

Kite Connect's official flow requires a human to click through a browser
login (user_id + password + 2FA) and land back on a redirect URL carrying
a `request_token`, which is then exchanged for an `access_token` that is
valid only until ~6am IST the next day. There is no officially supported
headless/API login - this module automates the *browser* login flow by
directly calling Zerodha's own login endpoints (the same ones the login
page's JavaScript calls), which is the standard pattern used across the
retail algo-trading community for exactly this purpose.

This is inherently more fragile than the rest of this codebase: it
depends on undocumented endpoints that Zerodha could change without
notice, and it has NOT been exercised against a real account (no live
Kite credentials were available while writing it). Before relying on
this in an unattended pipeline, run `generate_access_token()` manually
once against your real account and confirm it returns a working token.
"""

import re
from typing import Optional

import pyotp
import requests
import structlog
from kiteconnect import KiteConnect

logger = structlog.get_logger("astra_trade.kite_auth")

_LOGIN_URL = "https://kite.zerodha.com/api/login"
_TWOFA_URL = "https://kite.zerodha.com/api/twofa"
_CONNECT_URL = "https://kite.zerodha.com/connect/login"


class KiteLoginError(RuntimeError):
    """Raised when the automated Kite login flow fails at any step."""


def _extract_request_token(session: requests.Session, api_key: str) -> str:
    """
    Follow the connect/login redirect chain (without letting `requests`
    try to actually load the app's registered redirect URL, which may
    not be reachable from this environment) and pull `request_token`
    out of whichever Location header carries it.
    """
    url = f"{_CONNECT_URL}?api_key={api_key}&v=3"
    for _ in range(5):
        response = session.get(url, allow_redirects=False, timeout=15)
        location = response.headers.get("Location")
        if not location:
            break
        match = re.search(r"request_token=([^&]+)", location)
        if match:
            return match.group(1)
        url = location

    raise KiteLoginError(
        "Could not find request_token in the connect/login redirect chain. "
        "Zerodha may have changed their login flow, or the account/app "
        "configuration is invalid."
    )


def generate_access_token(
    api_key: str,
    api_secret: str,
    user_id: str,
    password: str,
    totp_secret: str,
) -> str:
    """
    Perform a full headless Kite login and return a fresh access_token.

    Args:
        api_key: Kite Connect app API key
        api_secret: Kite Connect app API secret
        user_id: Zerodha login ID (e.g. "AB1234")
        password: Zerodha account password
        totp_secret: Base32 TOTP seed (from the authenticator app setup QR
            code, not a 6-digit code) used to compute the current 2FA code

    Returns:
        A valid access_token, usable until Zerodha's daily session expiry.

    Raises:
        KiteLoginError: if any step of the login flow fails.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    login_resp = session.post(
        _LOGIN_URL, data={"user_id": user_id, "password": password}, timeout=15
    )
    login_data = login_resp.json()
    if login_data.get("status") != "success":
        raise KiteLoginError(f"Kite login step failed: {login_data}")
    request_id = login_data["data"]["request_id"]

    totp_code = pyotp.TOTP(totp_secret).now()
    twofa_resp = session.post(
        _TWOFA_URL,
        data={
            "user_id": user_id,
            "request_id": request_id,
            "twofa_value": totp_code,
            "twofa_type": "totp",
        },
        timeout=15,
    )
    twofa_data = twofa_resp.json()
    if twofa_data.get("status") != "success":
        raise KiteLoginError(f"Kite 2FA step failed: {twofa_data}")

    request_token = _extract_request_token(session, api_key)

    kite = KiteConnect(api_key=api_key)
    session_data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = session_data["access_token"]

    logger.info("kite_login_success", user_id=user_id)
    return access_token


def generate_access_token_from_env(env: Optional[dict] = None) -> str:
    """
    Convenience wrapper reading KITE_API_KEY, KITE_API_SECRET, KITE_USER_ID,
    KITE_PASSWORD, KITE_TOTP_SECRET from the environment (or a provided
    dict, e.g. for testing) and returning a fresh access_token.
    """
    import os

    source = env if env is not None else os.environ
    required = ["KITE_API_KEY", "KITE_API_SECRET", "KITE_USER_ID", "KITE_PASSWORD", "KITE_TOTP_SECRET"]
    missing = [key for key in required if not source.get(key)]
    if missing:
        raise KiteLoginError(f"Missing required environment variables: {missing}")

    return generate_access_token(
        api_key=source["KITE_API_KEY"],
        api_secret=source["KITE_API_SECRET"],
        user_id=source["KITE_USER_ID"],
        password=source["KITE_PASSWORD"],
        totp_secret=source["KITE_TOTP_SECRET"],
    )


if __name__ == "__main__":
    # Manual smoke test: `python3 -m src.utils.kite_auth` with the KITE_*
    # env vars set. Prints only whether it succeeded - never the token
    # itself, since this may run in a logged CI context.
    try:
        token = generate_access_token_from_env()
        print(f"Login succeeded. Access token length: {len(token)}")
    except KiteLoginError as e:
        print(f"Login failed: {e}")
        raise
