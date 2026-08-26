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

The login flow (4 steps):
  1. POST /api/login          user_id + password        -> request_id
  2. POST /api/twofa          request_id + TOTP         -> session cookies
  3. GET  /connect/login      api_key (follow redirects) -> request_token
  4. POST /session/token      checksum(key+token+secret) -> access_token

Step 3 is the fragile one: the redirect chain passes through
/connect/finish (intermediate, no token), may stop at /connect/authorize
(one-time consent screen), and eventually redirects to the app's
registered redirect URL with request_token in the query string. The
registered URL is typically http://127.0.0.1/ where nothing listens, so
we must capture the token from the Location header before following
that final hop.
"""

import re
import time
import urllib.parse
from typing import List, Optional

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


def _totp_with_headroom(secret: str, min_seconds: float = 5.0) -> str:
    """Generate a TOTP code, waiting if the current one is about to expire.

    A code generated with 1-2 seconds of validity left often expires
    before Zerodha's server validates it, causing a 2FA rejection that
    looks like a wrong seed. This waits for the next code when the
    current one has less than `min_seconds` of life remaining.
    """
    totp = pyotp.TOTP(secret)
    remaining = totp.interval - (time.time() % totp.interval)
    if remaining < min_seconds:
        wait = remaining + 0.5
        logger.info("totp_waiting_for_fresh_code", wait_seconds=round(wait, 1))
        time.sleep(wait)
    return totp.now()


def _extract_request_token(session: requests.Session, api_key: str) -> str:
    """Follow the /connect/login redirect chain and extract request_token.

    Three failure modes this handles:

    1. The final redirect target (the app's registered URL, usually
       127.0.0.1) is unreachable — we stop before following it.
    2. /connect/authorize appears — the app hasn't been authorised for
       this account yet (one-time manual approval needed).
    3. The token is embedded in a JS redirect in the page body rather
       than a Location header.
    """
    url = f"{_CONNECT_URL}?api_key={api_key}&v=3"
    seen_urls: List[str] = []
    body = ""

    for _ in range(10):
        try:
            response = session.get(url, allow_redirects=False, timeout=15)
        except requests.RequestException:
            break

        location = response.headers.get("Location", "")

        if location:
            seen_urls.append(location)
            match = re.search(r"request_token=([A-Za-z0-9]+)", location)
            if match:
                return match.group(1)
            url = location
            continue

        # No Location header — we've landed on a page. Check its body.
        body = response.text[:200_000]

        # Check for token in JS redirect or meta refresh
        match = re.search(r"request_token=([A-Za-z0-9]+)", body)
        if match:
            return match.group(1)

        break

    # Detect the consent screen by path, not page content (it's a JS
    # shell whose HTML doesn't contain "authorize" as visible text).
    landed = url
    all_urls = seen_urls + [landed]
    if any("/connect/authorize" in u for u in all_urls):
        raise KiteLoginError(
            "The Kite app has not been authorised for this account yet. "
            "Kite stopped at its consent screen (/connect/authorize), which "
            "needs ONE manual approval and then never appears again. "
            "Open https://kite.zerodha.com/connect/login?v=3&api_key="
            f"{api_key} in a browser, sign in, press Authorise, and re-run. "
            "The page failing to load AFTER you press Authorise is expected "
            "and means it worked."
        )

    # Strip query strings for safe logging (they can carry tokens)
    safe_hops = []
    for u in seen_urls:
        parsed = urllib.parse.urlparse(u)
        safe_hops.append(urllib.parse.urlunparse(
            parsed._replace(query="", fragment="")))

    raise KiteLoginError(
        "Could not find request_token in the connect/login redirect chain. "
        f"Redirect hops: {' -> '.join(safe_hops) or '(no redirects)'}. "
        "Possible causes: (1) the app's redirect URL on the Kite developer "
        "console doesn't match, (2) the TOTP seed is for a different account, "
        "(3) the password changed, or (4) Zerodha changed their login flow."
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

    # Step 1: password login
    logger.info("kite_login_step1", user_id=user_id)
    login_resp = session.post(
        _LOGIN_URL, data={"user_id": user_id, "password": password}, timeout=15
    )
    login_data = login_resp.json()
    if login_data.get("status") != "success":
        raise KiteLoginError(f"Kite login step failed: {login_data}")
    request_id = login_data["data"]["request_id"]

    # Step 2: TOTP with headroom to avoid near-expiry race
    totp_code = _totp_with_headroom(totp_secret)
    logger.info("kite_login_step2_twofa")
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
    if str(twofa_data.get("status", "success")).lower() == "error":
        raise KiteLoginError(
            f"Kite 2FA rejected: {twofa_data.get('message', twofa_data)}. "
            "The TOTP seed is most likely for a different account, or the "
            "server clock has drifted more than 30s."
        )
    if twofa_data.get("status") != "success":
        raise KiteLoginError(f"Kite 2FA step failed: {twofa_data}")

    # Step 3: collect request_token from redirect chain
    logger.info("kite_login_step3_request_token")
    request_token = _extract_request_token(session, api_key)

    # Step 4: exchange for access_token
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
    try:
        token = generate_access_token_from_env()
        print(f"Login succeeded. Access token length: {len(token)}")
    except KiteLoginError as e:
        print(f"Login failed: {e}")
        raise
