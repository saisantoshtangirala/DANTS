import numpy as np
import pandas as pd

from src.training.fii_dii_flow_features import build_institutional_flow_feature_panel


class TestBuildInstitutionalFlowFeaturePanel:
    def test_empty_input_returns_empty(self):
        assert build_institutional_flow_feature_panel(pd.DataFrame()).empty

    def test_builds_one_diff_column_per_column_x_lookback(self):
        idx = pd.bdate_range("2022-01-01", periods=30)
        wide = pd.DataFrame(
            {"dii_net_index_future": np.arange(30, dtype=float), "fii_net_index_future": np.arange(30, dtype=float) * 2},
            index=idx,
        )
        panel = build_institutional_flow_feature_panel(wide, lookbacks=(1, 5))
        expected_cols = {
            "dii_net_index_future_diff1", "dii_net_index_future_diff5",
            "fii_net_index_future_diff1", "fii_net_index_future_diff5",
        }
        assert set(panel.columns) == expected_cols
        assert len(panel) == 30

    def test_diff_values_are_causal_and_correct(self):
        idx = pd.bdate_range("2022-01-01", periods=10)
        wide = pd.DataFrame({"dii_net_index_future": np.arange(10, dtype=float)}, index=idx)
        panel = build_institutional_flow_feature_panel(wide, lookbacks=(3,))
        # diff3 at position i should equal value[i] - value[i-3], NaN for i < 3
        assert panel["dii_net_index_future_diff3"].iloc[:3].isna().all()
        assert panel["dii_net_index_future_diff3"].iloc[5] == 5.0 - 2.0

    def test_mutating_a_later_row_does_not_change_earlier_diffs(self):
        """Causal property: diff feature at day t must not depend on any
        value strictly after day t."""
        idx = pd.bdate_range("2022-01-01", periods=20)
        base = pd.Series(np.random.default_rng(0).normal(0, 1, 20))
        wide_before = pd.DataFrame({"dii_net_index_future": base.to_numpy()}, index=idx)
        panel_before = build_institutional_flow_feature_panel(wide_before, lookbacks=(2,))

        mutated = base.copy()
        mutated.iloc[15:] += 1000.0
        wide_after = pd.DataFrame({"dii_net_index_future": mutated.to_numpy()}, index=idx)
        panel_after = build_institutional_flow_feature_panel(wide_after, lookbacks=(2,))

        col = "dii_net_index_future_diff2"
        pd.testing.assert_series_equal(
            panel_before[col].iloc[:14], panel_after[col].iloc[:14], check_names=False,
        )
