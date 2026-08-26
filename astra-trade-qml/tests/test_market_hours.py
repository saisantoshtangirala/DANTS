from datetime import datetime
from zoneinfo import ZoneInfo

from src.trading.market_hours import is_market_open

SCHEDULE = {
    "pre_market": "09:00",
    "market_open": "09:15",
    "market_close": "15:30",
    "post_market": "15:30",
}
IST = ZoneInfo("Asia/Kolkata")


def test_open_during_market_hours():
    now = datetime(2026, 8, 26, 11, 0, tzinfo=IST)  # Wednesday, 11:00 IST
    assert is_market_open(SCHEDULE, now=now) is True


def test_closed_before_market_open():
    now = datetime(2026, 8, 26, 9, 0, tzinfo=IST)  # 09:00 IST, before 09:15 open
    assert is_market_open(SCHEDULE, now=now) is False


def test_closed_after_market_close():
    now = datetime(2026, 8, 26, 16, 0, tzinfo=IST)  # 16:00 IST, after 15:30 close
    assert is_market_open(SCHEDULE, now=now) is False


def test_closed_on_saturday():
    now = datetime(2026, 8, 29, 11, 0, tzinfo=IST)  # Saturday
    assert is_market_open(SCHEDULE, now=now) is False


def test_closed_on_sunday():
    now = datetime(2026, 8, 30, 11, 0, tzinfo=IST)  # Sunday
    assert is_market_open(SCHEDULE, now=now) is False


def test_open_exactly_at_boundaries():
    open_boundary = datetime(2026, 8, 26, 9, 15, tzinfo=IST)
    close_boundary = datetime(2026, 8, 26, 15, 30, tzinfo=IST)
    assert is_market_open(SCHEDULE, now=open_boundary) is True
    assert is_market_open(SCHEDULE, now=close_boundary) is True


def test_converts_from_other_timezones_to_ist():
    # UTC 06:00 = IST 11:30, well within market hours regardless of host TZ.
    utc = ZoneInfo("UTC")
    now = datetime(2026, 8, 26, 6, 0, tzinfo=utc)
    assert is_market_open(SCHEDULE, now=now) is True
