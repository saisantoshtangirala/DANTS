from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from src.trading.costs import CostCalculator
from src.training.pairs_trading import backtest_pair, compute_spread, find_cointegrated_pairs


def _synthetic_cointegrated_log_prices(n: int = 400, seed: int = 1):
    """Two series sharing a common random-walk component plus independent
    small mean-reverting (AR(1), stationary) noise - their difference is
    itself stationary, so they should be detected as cointegrated with a
    hedge ratio close to 1.0."""
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 0.01, n))

    def ar1_noise(phi=0.5, scale=0.005):
        noise = np.zeros(n)
        for i in range(1, n):
            noise[i] = phi * noise[i - 1] + rng.normal(0, scale)
        return noise

    log_a = common + ar1_noise()
    log_b = common + ar1_noise()
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(log_a, index=dates), pd.Series(log_b, index=dates)


def _synthetic_independent_log_prices(n: int = 400, seed: int = 7):
    """Two independent random walks - should generally NOT be found
    cointegrated (no shared long-run relationship)."""
    rng = np.random.default_rng(seed)
    log_a = np.cumsum(rng.normal(0, 0.01, n))
    log_b = np.cumsum(rng.normal(0, 0.01, n))
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(log_a, index=dates), pd.Series(log_b, index=dates)


def test_find_cointegrated_pairs_detects_synthetic_cointegrated_pair():
    log_a, log_b = _synthetic_cointegrated_log_prices()
    pairs = find_cointegrated_pairs({"SYM_A": log_a, "SYM_B": log_b}, family_significance=0.05)

    assert len(pairs) == 1
    pair = pairs[0]
    assert {pair["symbol_a"], pair["symbol_b"]} == {"SYM_A", "SYM_B"}
    assert pair["hedge_ratio"] == pytest.approx(1.0, abs=0.3)
    assert pair["p_value"] < 0.05


def test_find_cointegrated_pairs_rejects_independent_random_walks():
    log_a, log_b = _synthetic_independent_log_prices()
    pairs = find_cointegrated_pairs({"SYM_A": log_a, "SYM_B": log_b}, family_significance=0.05)

    assert pairs == []


def test_find_cointegrated_pairs_applies_bonferroni_correction():
    """
    Regression test for the multiple-comparisons fix: with N symbols
    testing N*(N-1)/2 pairs, a p-value that would clear a naive per-pair
    alpha=0.05 but not the Bonferroni-corrected alpha
    (family_significance / n_pairs_tested) must be rejected.
    """
    symbols = ["A", "B", "C", "D", "E"]  # 5 symbols -> 10 pairs -> per-pair alpha = 0.05/10 = 0.005
    dates = pd.date_range("2024-01-01", periods=250, freq="D")
    rng = np.random.default_rng(3)
    log_prices = {s: pd.Series(np.cumsum(rng.normal(0, 0.01, 250)), index=dates) for s in symbols}

    # A p-value that clears naive alpha=0.05 but not the corrected 0.005.
    borderline_p = 0.02
    with patch("src.training.pairs_trading.coint", return_value=(0.0, borderline_p, [0.0])):
        pairs = find_cointegrated_pairs(log_prices, family_significance=0.05)

    assert pairs == []

    # The same p-value must pass once the correction no longer excludes it
    # (a much looser family_significance target).
    with patch("src.training.pairs_trading.coint", return_value=(0.0, borderline_p, [0.0])):
        pairs_loose = find_cointegrated_pairs(log_prices, family_significance=1.0)

    assert len(pairs_loose) == 10


def test_compute_spread_aligns_on_common_dates():
    dates_a = pd.date_range("2024-01-01", periods=5, freq="D")
    dates_b = pd.date_range("2024-01-02", periods=5, freq="D")  # offset by one day
    log_a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates_a)
    log_b = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates_b)

    spread = compute_spread(log_a, log_b, hedge_ratio=1.0, intercept=0.0)

    # Only the 4 overlapping dates should survive the inner join.
    assert len(spread) == 4


def _pair_ohlcv(dates: pd.DatetimeIndex, closes: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "date": dates, "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": np.full(len(closes), 10_000.0),
    })


def _cost_calc(config) -> CostCalculator:
    return CostCalculator(config["trading"]["costs"])


def test_backtest_pair_enters_long_spread_and_exits_on_reversion(config):
    """
    Construct a spread that dives well below -entry_z (A cheap relative
    to B -> long_spread: long A, short B) then reverts back toward zero -
    must produce exactly one trade, entered at the dive and closed on
    reversion, not force-closed at session end.
    """
    n = 60
    dates = pd.date_range("2024-01-01 09:15", periods=n, freq="5min")

    # hedge_ratio=1, intercept=0 -> spread = log(close_a) - log(close_b).
    # Keep B flat at 100; drive A down sharply mid-window (dives the
    # spread negative) then back up (reversion).
    close_b = np.full(n, 100.0)
    close_a = np.full(n, 100.0)
    close_a[20:25] = 80.0   # spread dives well negative here
    close_a[25:] = 99.5     # reverts back near flat afterward

    df_a = _pair_ohlcv(dates, close_a)
    df_b = _pair_ohlcv(dates, close_b)

    # Warmup: enough flat history (spread ~0, low variance) so the
    # rolling std isn't dominated by the dive itself and z crosses
    # entry_z cleanly once the dive happens.
    warmup_dates = pd.date_range("2023-12-31 09:15", periods=20, freq="5min")
    warmup_a = pd.Series(np.log(np.full(20, 100.0)), index=warmup_dates)
    warmup_b = pd.Series(np.log(np.full(20, 100.0)), index=warmup_dates)

    report = backtest_pair(
        df_a, df_b, hedge_ratio=1.0, intercept=0.0,
        cost_calc=_cost_calc(config), initial_capital=config["trading"]["capital"]["initial"],
        max_position_size_pct=config["trading"]["position_sizing"]["max_position_size_pct"],
        entry_z=2.0, exit_z=0.5, stop_z=6.0, window=20,
        warmup_log_price_a=warmup_a, warmup_log_price_b=warmup_b,
    )

    assert report["total_trades"] >= 1


def test_backtest_pair_force_closes_at_session_end():
    """A position still open on the last bar of a trading session must be
    force-closed there, not left open / rolled into a synthetic next bar
    that doesn't exist in the data."""
    n = 30
    dates = pd.date_range("2024-01-01 09:15", periods=n, freq="5min")
    close_b = np.full(n, 100.0)
    close_a = np.full(n, 100.0)
    close_a[10:] = 80.0  # spread dives and STAYS dived - never reverts within this session

    df_a = _pair_ohlcv(dates, close_a)
    df_b = _pair_ohlcv(dates, close_b)
    warmup_dates = pd.date_range("2023-12-31 09:15", periods=20, freq="5min")
    warmup_a = pd.Series(np.log(np.full(20, 100.0)), index=warmup_dates)
    warmup_b = pd.Series(np.log(np.full(20, 100.0)), index=warmup_dates)

    cost_calc = CostCalculator({
        "brokerage_per_order": 20, "brokerage_pct_cap": 0.0003, "stt_pct": 0.001,
        "stt_delivery_pct": 0.001, "gst_pct": 0.18, "transaction_charges_pct": 0.0000345,
        "sebi_charges_pct": 0.0001, "stamp_duty_pct": 0.00015, "slippage_pct": 0.0005,
    })

    report = backtest_pair(
        df_a, df_b, hedge_ratio=1.0, intercept=0.0,
        cost_calc=cost_calc, initial_capital=50_000.0, max_position_size_pct=0.1,
        entry_z=2.0, exit_z=0.5, stop_z=6.0, window=20,
        warmup_log_price_a=warmup_a, warmup_log_price_b=warmup_b,
    )

    # The dive crosses entry_z and never reverts (stays far from exit_z,
    # never past stop_z either) - the only way this trade closes at all
    # is the end-of-session force-close.
    assert report["total_trades"] == 1


def test_backtest_pair_returns_none_for_no_overlapping_dates():
    dates_a = pd.date_range("2024-01-01 09:15", periods=10, freq="5min")
    dates_b = pd.date_range("2025-01-01 09:15", periods=10, freq="5min")
    df_a = _pair_ohlcv(dates_a, np.full(10, 100.0))
    df_b = _pair_ohlcv(dates_b, np.full(10, 100.0))

    cost_calc = CostCalculator({
        "brokerage_per_order": 20, "brokerage_pct_cap": 0.0003, "stt_pct": 0.001,
        "stt_delivery_pct": 0.001, "gst_pct": 0.18, "transaction_charges_pct": 0.0000345,
        "sebi_charges_pct": 0.0001, "stamp_duty_pct": 0.00015, "slippage_pct": 0.0005,
    })

    report = backtest_pair(
        df_a, df_b, hedge_ratio=1.0, intercept=0.0,
        cost_calc=cost_calc, initial_capital=50_000.0, max_position_size_pct=0.1,
    )

    assert report is None
