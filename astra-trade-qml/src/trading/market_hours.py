"""Market-hours gating for the paper trading loop, using config.yaml's
`trading.schedule` block. Always evaluates in IST regardless of the host
machine's local timezone (Hetzner VPS hosts default to UTC)."""

from datetime import datetime, time as dt_time
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo


def is_market_open(
    schedule_cfg: Dict[str, Any],
    now: Optional[datetime] = None,
    timezone: str = "Asia/Kolkata",
) -> bool:
    """
    True if `now` (default: current time) falls within NSE trading hours
    on a weekday, per schedule_cfg's market_open/market_close strings
    (e.g. "09:15", "15:30"). Does not account for market holidays.
    """
    tz = ZoneInfo(timezone)
    current = now.astimezone(tz) if now is not None else datetime.now(tz)

    if current.weekday() >= 5:  # Saturday, Sunday
        return False

    open_time = parse_hhmm(schedule_cfg["market_open"])
    close_time = parse_hhmm(schedule_cfg["market_close"])

    return open_time <= current.time() <= close_time


def parse_hhmm(value: str) -> dt_time:
    """Parse an "HH:MM" config string (e.g. square_off_time, no_new_entry_after)."""
    hour, minute = (int(x) for x in value.split(":"))
    return dt_time(hour=hour, minute=minute)
