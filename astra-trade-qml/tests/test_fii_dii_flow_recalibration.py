import numpy as np
import pandas as pd
import pytest

from src.trading.costs import CostCalculator
from src.training.fii_dii_flow_recalibration import (
    _ic_scan,
    is_recalibration_due,
    load_recalibration_state,
    run_fii_dii_flow_recalibration_check,
    save_recalibration_state,
)

HOLD_DAYS = 5
NOISE_COLUMNS = ("fii_net_index_future", "pro_net_index_future", "client_net_index_future")


@pytest.fixture
def cost_calc():
    return CostCalculator({})


def _forward_return_label(closes: np.ndarray, hold_days: int) -> np.ndarray:
    n = len(closes)
    label = np.full(n, np.nan)
    for t in range(max(0, n - hold_days - 1)):
        entry = closes[t + 1]
        exitp = closes[t + 1 + hold_days]
        label[t] = (exitp - entry) / entry
    return label


def _planted_signal_dataset(n: int, beta: float, noise_scale: float, seed: int):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)

    dii_level = np.cumsum(rng.normal(0, 100, n))
    dii_diff5 = pd.Series(dii_level).diff(5).to_numpy()
    dii_diff5_filled = np.nan_to_num(dii_diff5, nan=0.0)
    dii_diff5_norm = dii_diff5_filled / (np.std(dii_diff5_filled[20:]) + 1e-9)

    daily_log_return = np.zeros(n)
    for k in range(n):
        src_idx = max(0, k - 1)
        daily_log_return[k] = (beta / HOLD_DAYS) * dii_diff5_norm[src_idx] + rng.normal(0, noise_scale)
    close = 100 * np.exp(np.cumsum(daily_log_return))

    price_df = pd.DataFrame({"date": dates, "close": close})
    wide = pd.DataFrame({"dii_net_index_future": dii_level}, index=dates)
    for col in NOISE_COLUMNS:
        wide[col] = np.cumsum(rng.normal(0, 100, n))

    return price_df, wide


def _pure_noise_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    price_df = pd.DataFrame({"date": dates, "close": close})
    cols = ("dii_net_index_future",) + NOISE_COLUMNS
    wide = pd.DataFrame({c: np.cumsum(rng.normal(0, 100, n)) for c in cols}, index=dates)
    return price_df, wide


class TestRecalibrationStatePersistence:
    def test_round_trip(self, tmp_path):
        state_path = str(tmp_path / "state.json")
        verdict = {"recommendation": "no change needed", "current_feature_still_valid": True}
        save_recalibration_state(state_path, verdict)

        loaded = load_recalibration_state(state_path)
        assert loaded is not None
        assert loaded["verdict"] == verdict
        assert "last_run" in loaded

    def test_load_missing_file_returns_none(self, tmp_path):
        assert load_recalibration_state(str(tmp_path / "does_not_exist.json")) is None

    def test_load_corrupt_file_returns_none(self, tmp_path):
        state_path = tmp_path / "corrupt.json"
        state_path.write_text("{not valid json")
        assert load_recalibration_state(str(state_path)) is None


class TestIsRecalibrationDue:
    def test_true_when_never_run(self, tmp_path):
        assert is_recalibration_due(str(tmp_path / "state.json"), frequency_days=30) is True

    def test_false_when_recently_run(self, tmp_path):
        state_path = str(tmp_path / "state.json")
        save_recalibration_state(state_path, {"recommendation": "ok"})
        assert is_recalibration_due(state_path, frequency_days=30) is False

    def test_true_when_stale(self, tmp_path):
        import json as _json
        from datetime import datetime, timedelta

        state_path = tmp_path / "state.json"
        stale_state = {
            "last_run": (datetime.now() - timedelta(days=45)).isoformat(),
            "verdict": {"recommendation": "ok"},
        }
        state_path.write_text(_json.dumps(stale_state))
        assert is_recalibration_due(str(state_path), frequency_days=30) is True


class TestIcScan:
    def test_identifies_planted_signal_as_passing_and_top_ranked(self):
        price_df, wide = _planted_signal_dataset(n=700, beta=0.08, noise_scale=0.01, seed=1)
        result = _ic_scan(price_df, wide, hold_days=HOLD_DAYS, lookbacks=(3, 5, 10, 20), alpha=0.05)

        assert result["n_candidates_tested"] > 0
        assert len(result["passing_candidates"]) > 0
        top = result["candidates"][0]
        assert "dii_net_index_future" in top["feature"]
        assert top["passes_bonferroni_both_halves"] is True

    def test_rejects_pure_noise(self):
        price_df, wide = _pure_noise_dataset(n=700, seed=2)
        result = _ic_scan(price_df, wide, hold_days=HOLD_DAYS, lookbacks=(3, 5, 10, 20), alpha=0.05)

        assert result["n_candidates_tested"] > 0
        assert len(result["passing_candidates"]) == 0


class TestRunFiiDiiFlowRecalibrationCheck:
    def test_end_to_end_confirms_valid_feature_still_passes(self, cost_calc):
        price_df, wide = _planted_signal_dataset(n=700, beta=0.08, noise_scale=0.01, seed=3)

        result = run_fii_dii_flow_recalibration_check(
            price_df, wide, cost_calc, initial_capital=50000.0,
            hold_days=HOLD_DAYS, max_concurrent_positions=5, n_folds=3,
            current_feature="dii_net_index_future", current_lookback=5,
        )

        assert "ic_scan" in result
        assert "classical_ml_result" in result
        assert result["pooled_ml_result"] is None
        verdict = result["verdict"]
        assert verdict["current_feature_still_valid"] is True
        assert verdict["current_candidate"] == "dii_net_index_future_diff5"
        assert "no change needed" in verdict["recommendation"] or "beats the rule-based benchmark" in verdict["recommendation"]

    def test_end_to_end_flags_stale_feature_on_pure_noise(self, cost_calc):
        price_df, wide = _pure_noise_dataset(n=700, seed=4)

        result = run_fii_dii_flow_recalibration_check(
            price_df, wide, cost_calc, initial_capital=50000.0,
            hold_days=HOLD_DAYS, max_concurrent_positions=5, n_folds=3,
            current_feature="dii_net_index_future", current_lookback=5,
        )

        verdict = result["verdict"]
        assert verdict["current_feature_still_valid"] is False
        assert verdict["best_candidate"] is None
