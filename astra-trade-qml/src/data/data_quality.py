"""
Data-quality safeguards for the trading universe.

The "survivorship bias" risk the Indian-market algo guide warns about is
mainly about backtesting only on symbols that are CURRENTLY in an index,
silently dropping symbols that were removed or delisted along the way.
This system doesn't select its universe dynamically from a historical
index list - it trades a fixed, hand-picked set of large caps
(RELIANCE/TCS/HDFCBANK/ICICIBANK/SBIN/BHARTIARTL/ITC/INFY), all of which
have been continuously listed and Nifty50 constituents for the system's
entire relevant history. The classic form of the bias - "the backtest
only sees survivors because the delisted names were quietly dropped from
today's list" - barely applies to a fixed universe like this one.

What DOES apply, and generalizes if the universe is ever widened to
include less stable names, is silently trusting data from a symbol that
was actually halted, suspended, or thinly traded for part of the window
being backtested. That's what this module checks for.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class ListingContinuityReport:
    symbol: str
    coverage_ratio: float          # rows present / expected trading days in the window
    max_gap_days: float            # longest gap between consecutive trading days
    zero_volume_ratio: float       # fraction of bars with zero/negative volume
    is_continuous: bool
    reason: Optional[str] = None


def check_listing_continuity(
    symbol: str,
    df: pd.DataFrame,
    min_coverage_ratio: float = 0.90,
    max_gap_days: float = 5.0,
    max_zero_volume_ratio: float = 0.05,
) -> ListingContinuityReport:
    """
    Flag a symbol whose data suggests it wasn't continuously tradable
    across the window - a halt, suspension, or delisting - so a backtest
    or the capital allocator doesn't silently treat a data gap as "no
    signal that day" rather than "this symbol may not have been
    tradable."

    Args:
        symbol: Symbol name (for the report only)
        df: OHLCV DataFrame with "date" and "volume" columns spanning the window to check
        min_coverage_ratio: Minimum fraction of expected trading days with at least one bar present
        max_gap_days: Longest allowed gap between consecutive trading days before it's treated as a halt
        max_zero_volume_ratio: Maximum fraction of bars with zero volume before the symbol is flagged
    """
    if df.empty or "date" not in df.columns:
        return ListingContinuityReport(
            symbol=symbol, coverage_ratio=0.0, max_gap_days=float("inf"),
            zero_volume_ratio=1.0, is_continuous=False, reason="no data",
        )

    trading_days = pd.to_datetime(df["date"]).dt.normalize().drop_duplicates().sort_values()
    if len(trading_days) < 2:
        return ListingContinuityReport(
            symbol=symbol, coverage_ratio=0.0, max_gap_days=float("inf"),
            zero_volume_ratio=1.0, is_continuous=False, reason="fewer than 2 trading days present",
        )

    # Business-day calendar approximates NSE trading days (doesn't
    # account for exchange holidays, so real gaps read slightly smaller
    # than the calendar implies - conservative in the direction of not
    # over-flagging a symbol for holidays that aren't real gaps).
    expected_days = pd.bdate_range(trading_days.min(), trading_days.max())
    coverage_ratio = len(trading_days) / max(len(expected_days), 1)

    gaps = trading_days.diff().dt.days.dropna()
    max_gap = float(gaps.max()) if not gaps.empty else 0.0

    zero_volume_ratio = 0.0
    if "volume" in df.columns and len(df) > 0:
        zero_volume_ratio = float((df["volume"] <= 0).mean())

    reasons = []
    if coverage_ratio < min_coverage_ratio:
        reasons.append(f"coverage {coverage_ratio:.0%} below {min_coverage_ratio:.0%}")
    if max_gap > max_gap_days:
        reasons.append(f"longest gap {max_gap:.0f}d exceeds {max_gap_days:.0f}d")
    if zero_volume_ratio > max_zero_volume_ratio:
        reasons.append(f"{zero_volume_ratio:.0%} zero-volume bars exceeds {max_zero_volume_ratio:.0%}")

    return ListingContinuityReport(
        symbol=symbol,
        coverage_ratio=coverage_ratio,
        max_gap_days=max_gap,
        zero_volume_ratio=zero_volume_ratio,
        is_continuous=not reasons,
        reason="; ".join(reasons) if reasons else None,
    )
