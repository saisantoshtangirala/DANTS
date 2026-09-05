"""
Raw institutional-flow feature panel.

fii_dii_flow.py's validated rule uses exactly one hand-picked feature
(DII's 5-day change in net NIFTY index-futures open interest) because
that's the one the original exploratory IC scan (120 feature x horizon
combinations - see fii_dii_flow.py's module docstring) found and
confirmed. This module expands that single winning feature back out
into the fuller raw space NSE's participant-OI disclosure actually
carries - every (client_type, instrument) pair
compute_net_positioning() computes, at several lookback windows - so
fii_dii_flow_quantum.py's genetic feature evolution and quantum
classifier have real raw material to search over instead of assuming
the one already-validated feature is the only useful signal in this
data.
"""

from typing import Sequence

import pandas as pd

DEFAULT_LOOKBACKS = (3, 5, 10, 20)


def build_institutional_flow_feature_panel(
    net_positioning_wide: pd.DataFrame,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
) -> pd.DataFrame:
    """net_positioning_wide: date-indexed, one column per (client_type,
    instrument) pair - compute_net_positioning()'s output (e.g.
    "dii_net_index_future", "fii_net_stock_future",
    "pro_net_index_option_bias", ...).

    Returns a date-indexed DataFrame of causal diff features:
    "{column}_diff{lookback}" for every column x lookback combination.
    Each value at day t is net_positioning[t] - net_positioning[t -
    lookback] - uses only information through day t, no look-ahead.
    Empty input returns an empty DataFrame (matches
    compute_net_positioning's own empty-input convention).
    """
    if net_positioning_wide.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=net_positioning_wide.index)
    for col in net_positioning_wide.columns:
        for lb in lookbacks:
            out[f"{col}_diff{lb}"] = net_positioning_wide[col].diff(lb)
    return out
