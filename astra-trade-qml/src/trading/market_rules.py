"""NSE market-structure rules that a realistic backtest or paper fill must
respect: tick-size rounding and daily circuit (price band) limits.

Circuit bands are set per-security by the exchange (2/5/10/20%, or none for
the most liquid F&O-enabled large caps) and can change over time. We don't
have a feed for the actual per-symbol band, so this uses a configurable
default band as a conservative approximation good enough to catch the
clearly-implausible fills a naive backtest would otherwise allow (e.g.
"buying" into a move larger than any real circuit would have permitted).
"""

from dataclasses import dataclass

TICK_SIZE = 0.05


def round_to_tick(price: float, tick_size: float = TICK_SIZE) -> float:
    """Round a price to the nearest valid NSE tick (default ₹0.05)."""
    if tick_size <= 0:
        return price
    return round(round(price / tick_size) * tick_size, 2)


@dataclass
class CircuitCheck:
    band_pct: float = 0.20

    def is_frozen(self, prev_close: float, price: float) -> bool:
        """True if `price` implies a move beyond the daily circuit band from
        `prev_close`, meaning a real order at that price likely wouldn't
        have filled (the stock would be locked at the limit with one-sided
        depth)."""
        if prev_close <= 0:
            return False
        move_pct = abs(price - prev_close) / prev_close
        return move_pct >= self.band_pct

    def would_fill(self, prev_close: float, bar_high: float, bar_low: float) -> bool:
        """True if a bar's high/low range stays within the circuit band of
        the previous close, i.e. this bar wasn't frozen at a circuit limit
        for its entire range."""
        if prev_close <= 0:
            return True
        upper = prev_close * (1 + self.band_pct)
        lower = prev_close * (1 - self.band_pct)
        return not (bar_low >= upper or bar_high <= lower)
