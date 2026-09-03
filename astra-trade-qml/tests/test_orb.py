from datetime import date, datetime, time

import numpy as np
import pandas as pd
import pytest

from src.trading.costs import CostCalculator
from src.training.orb import (
    compute_opening_range,
    find_first_breakout,
    run_orb_backtest,
    simulate_orb_trade,
)

MARKET_OPEN = time(9, 15)


def _day_bars(day: date, bars):
    """bars: list of (minutes_after_open, open, high, low, close, volume)."""
    rows = []
    for minutes, o, h, l, c, v in bars:
        ts = datetime.combine(day, MARKET_OPEN) + pd.Timedelta(minutes=minutes)
        rows.append({"date": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
    return pd.DataFrame(rows)


@pytest.fixture
def cost_calc():
    return CostCalculator({})


class TestComputeOpeningRange:
    def test_computes_high_low_avg_volume_from_first_window(self):
        day = date(2024, 1, 2)
        bars = _day_bars(day, [
            (0, 100, 102, 99, 101, 1000),   # 09:15
            (5, 101, 103, 100, 102, 1200),  # 09:20
            (10, 102, 104, 101, 103, 800),  # 09:25
            (15, 103, 106, 102, 105, 900),  # 09:30 - outside a 15-min window
        ])
        result = compute_opening_range(bars, MARKET_OPEN, opening_range_minutes=15)
        assert result is not None
        range_high, range_low, avg_vol = result
        assert range_high == 104.0  # max high of the first 3 bars only
        assert range_low == 99.0
        assert avg_vol == pytest.approx((1000 + 1200 + 800) / 3)

    def test_none_with_too_few_bars(self):
        day = date(2024, 1, 2)
        bars = _day_bars(day, [(0, 100, 101, 99, 100, 500)])
        assert compute_opening_range(bars, MARKET_OPEN, opening_range_minutes=15) is None


class TestFindFirstBreakout:
    def _setup(self, post_range_bars):
        day = date(2024, 1, 2)
        opening = [
            (0, 100, 101, 99, 100, 1000),
            (5, 100, 101, 99, 100, 1000),
            (10, 100, 101, 99, 100, 1000),
        ]
        bars = _day_bars(day, opening + post_range_bars)
        return bars, 101.0, 99.0, 1000.0  # range_high, range_low, avg_range_volume

    def test_detects_long_breakout_with_volume_confirmation(self):
        bars, rh, rl, avg_vol = self._setup([(15, 101, 103, 101, 102.5, 2000)])
        result = find_first_breakout(bars, rh, rl, avg_vol, MARKET_OPEN, 15, 1.5, time(15, 0))
        assert result is not None
        assert result["direction"] == "long"
        assert result["entry_price"] == pytest.approx(102.5)

    def test_detects_short_breakout(self):
        bars, rh, rl, avg_vol = self._setup([(15, 99, 99, 97, 97.5, 2000)])
        result = find_first_breakout(bars, rh, rl, avg_vol, MARKET_OPEN, 15, 1.5, time(15, 0))
        assert result is not None
        assert result["direction"] == "short"

    def test_ignores_breakout_without_volume_confirmation(self):
        bars, rh, rl, avg_vol = self._setup([(15, 101, 103, 101, 102.5, 900)])  # below 1.5x avg
        result = find_first_breakout(bars, rh, rl, avg_vol, MARKET_OPEN, 15, 1.5, time(15, 0))
        assert result is None

    def test_no_breakout_when_price_stays_in_range(self):
        bars, rh, rl, avg_vol = self._setup([(15, 100, 100.5, 99.5, 100, 2000)])
        result = find_first_breakout(bars, rh, rl, avg_vol, MARKET_OPEN, 15, 1.5, time(15, 0))
        assert result is None

    def test_respects_no_new_entry_cutoff(self):
        # Breakout bar is at 09:15 + 6 hours - well past a 10:00 cutoff.
        bars, rh, rl, avg_vol = self._setup([(360, 101, 103, 101, 102.5, 2000)])
        result = find_first_breakout(bars, rh, rl, avg_vol, MARKET_OPEN, 15, 1.5, time(10, 0))
        assert result is None

    def test_takes_first_breakout_only(self):
        bars, rh, rl, avg_vol = self._setup([
            (15, 101, 103, 101, 102.5, 2000),  # first: long breakout
            (20, 97, 97, 95, 95, 2000),         # second: would be short - ignored
        ])
        result = find_first_breakout(bars, rh, rl, avg_vol, MARKET_OPEN, 15, 1.5, time(15, 0))
        assert result["direction"] == "long"


class TestSimulateOrbTrade:
    def test_long_stop_hit(self):
        day = date(2024, 1, 2)
        bars = _day_bars(day, [
            (0, 100, 101, 99, 100, 1000), (5, 100, 101, 99, 100, 1000), (10, 100, 101, 99, 100, 1000),
            (15, 102, 103, 101.5, 102.5, 2000),  # entry bar (idx 3)
            (20, 102, 102.2, 97, 97.5, 2000),    # dips below stop (99)
        ])
        trade = simulate_orb_trade(bars, entry_idx=3, direction="long", entry_price=102.5,
                                    range_high=101.0, range_low=99.0, reward_multiple=2.0,
                                    square_off_time=time(15, 15))
        assert trade["exit_reason"] == "stop"
        assert trade["exit_price"] == pytest.approx(99.0)

    def test_long_target_hit(self):
        day = date(2024, 1, 2)
        # risk = 102.5 - 99 = 3.5, target = 102.5 + 2*3.5 = 109.5
        bars = _day_bars(day, [
            (0, 100, 101, 99, 100, 1000), (5, 100, 101, 99, 100, 1000), (10, 100, 101, 99, 100, 1000),
            (15, 102, 103, 101.5, 102.5, 2000),
            (20, 103, 110, 102, 109.8, 2000),
        ])
        trade = simulate_orb_trade(bars, entry_idx=3, direction="long", entry_price=102.5,
                                    range_high=101.0, range_low=99.0, reward_multiple=2.0,
                                    square_off_time=time(15, 15))
        assert trade["exit_reason"] == "target"
        assert trade["exit_price"] == pytest.approx(109.5)

    def test_short_stop_and_target_use_opposite_sides(self):
        day = date(2024, 1, 2)
        # short entry 97.5, stop = range_high = 101, risk = 3.5, target = 97.5 - 7 = 90.5
        bars = _day_bars(day, [
            (0, 100, 101, 99, 100, 1000), (5, 100, 101, 99, 100, 1000), (10, 100, 101, 99, 100, 1000),
            (15, 98, 98, 96, 97.5, 2000),
            (20, 97, 97, 90, 90.2, 2000),
        ])
        trade = simulate_orb_trade(bars, entry_idx=3, direction="short", entry_price=97.5,
                                    range_high=101.0, range_low=99.0, reward_multiple=2.0,
                                    square_off_time=time(15, 15))
        assert trade["exit_reason"] == "target"
        assert trade["exit_price"] == pytest.approx(90.5)

    def test_force_closes_at_square_off_when_neither_hit(self):
        day = date(2024, 1, 2)
        bars = _day_bars(day, [
            (0, 100, 101, 99, 100, 1000), (5, 100, 101, 99, 100, 1000), (10, 100, 101, 99, 100, 1000),
            (15, 102, 103, 101.5, 102.5, 2000),
            (355, 103, 103.5, 102.5, 103.2, 1500),  # 09:15+355min = 15:10, still before 15:15
        ])
        trade = simulate_orb_trade(bars, entry_idx=3, direction="long", entry_price=102.5,
                                    range_high=101.0, range_low=99.0, reward_multiple=2.0,
                                    square_off_time=time(15, 15))
        assert trade["exit_reason"] == "square_off"
        assert trade["exit_price"] == pytest.approx(103.2)

    def test_none_when_no_bars_after_entry(self):
        day = date(2024, 1, 2)
        bars = _day_bars(day, [
            (0, 100, 101, 99, 100, 1000), (5, 100, 101, 99, 100, 1000), (10, 100, 101, 99, 100, 1000),
            (15, 102, 103, 101.5, 102.5, 2000),
        ])
        trade = simulate_orb_trade(bars, entry_idx=3, direction="long", entry_price=102.5,
                                    range_high=101.0, range_low=99.0, reward_multiple=2.0,
                                    square_off_time=time(15, 15))
        assert trade is None


class TestRunOrbBacktest:
    def _multi_day_price_data(self, n_days=30, seed=0):
        rng = np.random.default_rng(seed)
        symbols = ["SYM_A", "SYM_B", "SYM_C"]
        price_data = {}
        for si, symbol in enumerate(symbols):
            rows = []
            price = 100.0 + si * 10
            for d in range(n_days):
                day = date(2024, 1, 1) + pd.Timedelta(days=d)
                if day.weekday() >= 5:
                    continue
                # opening range: 3 flat-ish bars
                o_h_l_c = price + rng.normal(0, 0.2, 3)
                for m in (0, 5, 10):
                    rows.append({
                        "date": datetime.combine(day, MARKET_OPEN) + pd.Timedelta(minutes=m),
                        "open": price, "high": price + 0.5, "low": price - 0.5, "close": price,
                        "volume": 1000.0,
                    })
                # occasional breakout with volume spike
                direction = rng.choice([-1, 0, 1])
                if direction != 0:
                    move = direction * rng.uniform(2, 5)
                    for m in (15, 20, 25, 30, 300):
                        rows.append({
                            "date": datetime.combine(day, MARKET_OPEN) + pd.Timedelta(minutes=m),
                            "open": price, "high": price + abs(move) + 1, "low": price - abs(move) - 1,
                            "close": price + move, "volume": 3000.0,
                        })
                price *= 1 + rng.normal(0.0002, 0.005)
            price_data[symbol] = pd.DataFrame(rows)
        return price_data

    def test_produces_trades_and_train_oos_split(self, cost_calc):
        price_data = self._multi_day_price_data()
        result = run_orb_backtest(
            price_data, cost_calc, initial_capital=50_000.0, max_position_size_pct=0.1,
        )
        assert result["n_days_with_data"] > 0
        assert result["n_trades"] > 0
        assert "train" in result and "oos" in result

    def test_no_trades_returns_degenerate_result(self, cost_calc):
        # Flat data all day, every day - no breakouts possible.
        day = date(2024, 1, 2)
        flat = _day_bars(day, [(m, 100, 100.1, 99.9, 100, 1000) for m in range(0, 370, 5)])
        result = run_orb_backtest(
            {"FLAT": flat}, cost_calc, initial_capital=50_000.0, max_position_size_pct=0.1,
        )
        assert result["n_trades"] == 0
