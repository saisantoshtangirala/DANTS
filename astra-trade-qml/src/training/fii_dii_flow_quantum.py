"""
Quantum-kernel classifier + genetic feature evolution over the FII/DII
institutional-flow raw feature space, walk-forward validated against
fii_dii_flow.py's single-feature rule-based signal - the one already
live in paper trading.

Why this exists: the rule-based signal (DII's 5-day net-index-future
OI change, top-quintile causal rank) is a single hand-picked feature
found by an exploratory IC scan and then frozen. This module asks
whether a model that (a) searches a much wider raw feature space
(src/training/fii_dii_flow_features.py - every client_type x
instrument x lookback combination, not just the one DII feature) and
(b) genetically evolves derived combinations of those features
(src/training/feature_evolution.py, already built and unit-tested for
this purpose) and (c) classifies with this codebase's existing quantum
kernel SVM (src/models/quantum/quantum_kernel.py, already validated
and production-hardened in the LSTM+XGBoost+quantum ensemble) can beat
the frozen rule out-of-sample.

Walk-forward design (no single train/test split - every reported trade
is genuinely out-of-sample by construction):

    Full history
    ├── warmup (min_train_days, causal feature lookback needs it)
    └── eligible range, split into `n_folds` expanding-window folds
        Fold i: train on everything strictly BEFORE the fold
                -> evolve genes (IC-scored on the train slice only)
                -> fit QuantumKernelClassifier on train slice
                -> decision threshold = top `quantile_threshold`
                   quantile of the FITTED MODEL's OWN predicted
                   probabilities on its train slice (a calibration
                   choice made entirely from information available
                   before the fold starts - not tuned on the fold's
                   own outcome)
                -> apply that fixed model + threshold to the fold's
                   dates only, producing this fold's entry signals

Every fold's entry signals get combined into one full-history
admissibility series and run through
fii_dii_flow.simulate_concurrent_tranche_trades - the EXACT SAME
execution engine (same 1-day publication lag, same hold_days exit,
same CostCalculator) the rule-based backtest uses, so the two
approaches' resulting OOS trade sets are genuinely apples-to-apples:
they differ only in which days pass the entry test, nothing about how
a passing day becomes a trade.

Adoption gate (compare_to_rule_based_benchmark): this module does NOT
decide for itself that it's better. It's only recommended for paper
trading if its own walk-forward OOS Sharpe ratio AND one-sample
significance both beat the rule-based benchmark - recomputed fresh
over the SAME eligible date range this run's folds cover (never a
stale hardcoded number from a previous run), per this project's
standing rule against ever fabricating or spinning a backtest result.
If it doesn't clear that bar, the documented fallback is periodic
walk-forward re-calibration of the existing rule's own parameters, not
replacing a validated simple rule with an unproven, more complex one.
Either way, this module never touches src/trading/fii_dii_flow_paper.py
directly - swapping paper trading over to a model this finds would be
a separate, explicit step taken only after a run here recommends it.
"""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.models.quantum.quantum_kernel import QuantumKernelClassifier
from src.trading.costs import CostCalculator
from src.training.feature_evolution import FeatureEvolver
from src.training.fii_dii_flow import compute_rolling_quantile_rank, simulate_concurrent_tranche_trades
from src.training.fii_dii_flow_features import DEFAULT_LOOKBACKS, build_institutional_flow_feature_panel
from src.training.fii_dii_flow_stress_test import one_sample_significance_test
from src.utils.metrics import generate_performance_report

MIN_TRAIN_DAYS = 252  # at least one year of history before the first fold


def _forward_return_label(closes: List[float], hold_days: int) -> np.ndarray:
    """label[t] = return of buying at day t+1's close and selling at day
    t+1+hold_days's close - exactly the trade fii_dii_flow.py's 1-day-lag
    execution rule would take if day t's signal fired. NaN wherever that
    forward window doesn't exist (the last hold_days+1 days)."""
    n = len(closes)
    label = np.full(n, np.nan)
    for t in range(max(0, n - hold_days - 1)):
        entry_price = closes[t + 1]
        exit_price = closes[t + 1 + hold_days]
        label[t] = (exit_price - entry_price) / entry_price if entry_price else np.nan
    return label


def _fold_boundaries(n_eligible: int, n_folds: int) -> List[Tuple[int, int]]:
    """Expanding-window fold OOS ranges over indices [0, n_eligible):
    fold i's OOS range is one contiguous chronological slice; fold i's
    training data is everything strictly before it. Same equal-width-
    OOS-fold convention as this codebase's swing_walk_forward_validation."""
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1.")
    boundaries = np.linspace(0, n_eligible, n_folds + 1, dtype=int)
    return [(int(boundaries[i]), int(boundaries[i + 1])) for i in range(n_folds)]


def _report_from_trades(trades_df: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
    if trades_df.empty:
        return {}
    equity_curve = (1 + trades_df["pnl_pct"]).cumprod() * initial_capital
    report = generate_performance_report(trades_df, equity_curve)
    # Same Sharpe-annualization fix as fii_dii_flow.py's _report(): each
    # "period" here is one trade, not one daily bar - see that module's
    # docstring for why the default 252-annualized figure is wrong.
    pnl_pct = trades_df["pnl_pct"].to_numpy()
    exit_dates = pd.to_datetime(trades_df["exit_date"])
    span_days = (exit_dates.iloc[-1] - exit_dates.iloc[0]).days
    trades_per_year = len(trades_df) / (span_days / 365.25) if span_days > 0 else float(len(trades_df))
    std = pnl_pct.std(ddof=1) if len(pnl_pct) > 1 else 0.0
    report["sharpe_ratio"] = float(pnl_pct.mean() / std * np.sqrt(trades_per_year)) if std > 1e-12 else 0.0
    report["trades_per_year"] = float(trades_per_year)
    report["period_returns"] = list(trades_df["pnl_pct"])
    report["period_dates"] = list(trades_df["exit_date"])
    return report


def _rule_based_benchmark_same_window(
    price_df: pd.DataFrame,
    net_positioning_wide: pd.DataFrame,
    cost_calc: CostCalculator,
    initial_capital: float,
    eligible_start: int,
    flow_lookback_days: int,
    trailing_window: int,
    quantile_threshold: float,
    hold_days: int,
    max_concurrent_positions: int,
) -> Dict[str, Any]:
    """Re-runs fii_dii_flow.py's validated single-feature rule,
    restricted to entries at or after `eligible_start` - the SAME date
    range run_fii_dii_flow_quantum_backtest's folds cover (not that
    rule's own 70/30 split from fii_dii_flow_backtest). A "does the
    quantum model beat the existing rule" verdict only means something
    if both are scored over identical dates."""
    price_df = price_df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    dates = price_df["date"].tolist()
    closes = price_df["close"].tolist()
    n = len(dates)

    feat = net_positioning_wide["dii_net_index_future"].diff(flow_lookback_days)
    quantile_rank = compute_rolling_quantile_rank(feat, trailing_window)

    entry_ok = pd.Series(False, index=range(n))
    for i, d in enumerate(dates):
        if i < eligible_start:
            continue
        qr = quantile_rank.get(d)
        if qr is not None and not pd.isna(qr) and qr >= quantile_threshold:
            entry_ok.iloc[i] = True

    position_notional = initial_capital / max_concurrent_positions
    trades = simulate_concurrent_tranche_trades(
        dates, closes, entry_ok, hold_days, max_concurrent_positions, cost_calc, position_notional,
    )
    if not trades:
        return {"n_trades": 0}

    trades_df = pd.DataFrame(trades).sort_values("exit_date").reset_index(drop=True)
    return _report_from_trades(trades_df, initial_capital)


def compare_to_rule_based_benchmark(quantum_oos: Dict[str, Any], benchmark: Dict[str, Any]) -> Dict[str, Any]:
    """Explicit adoption gate. The quantum/ML approach only counts as
    having beaten the validated rule-based signal if its walk-forward
    OOS Sharpe is higher AND its one-sample significance is at least as
    strong - a partial win (e.g. higher Sharpe but a weaker p-value) is
    reported as "does not beat benchmark", not spun as a win. See this
    module's docstring for the fallback if it doesn't clear this bar."""
    quantum_returns = quantum_oos.get("period_returns", [])
    quantum_sig = one_sample_significance_test(quantum_returns)
    quantum_sharpe = quantum_oos.get("sharpe_ratio", 0.0) or 0.0
    quantum_p = quantum_sig.get("p_value")

    benchmark_returns = benchmark.get("period_returns", [])
    benchmark_sig = one_sample_significance_test(benchmark_returns)
    benchmark_sharpe = benchmark.get("sharpe_ratio", 0.0) or 0.0
    benchmark_p = benchmark_sig.get("p_value")

    beats_sharpe = quantum_sharpe > benchmark_sharpe
    beats_significance = (
        quantum_p is not None and benchmark_p is not None and quantum_p <= benchmark_p
    )
    verdict = "beats_benchmark" if (beats_sharpe and beats_significance) else "does_not_beat_benchmark"

    return {
        "quantum_oos_sharpe": quantum_sharpe,
        "quantum_oos_p_value": quantum_p,
        "quantum_oos_n_trades": quantum_sig.get("n_trades", 0),
        "benchmark_oos_sharpe": benchmark_sharpe,
        "benchmark_oos_p_value": benchmark_p,
        "benchmark_oos_n_trades": benchmark_sig.get("n_trades", 0),
        "beats_sharpe": beats_sharpe,
        "beats_significance": beats_significance,
        "verdict": verdict,
        "recommendation": (
            "Adopt for paper trading - clears both the Sharpe and significance bar the "
            "validated rule-based signal set, over the same evaluation window."
            if verdict == "beats_benchmark"
            else "Keep the existing rule-based signal live. Fall back to periodic walk-forward "
            "re-calibration of its own parameters rather than replacing a validated simple rule "
            "with an unproven, more complex one."
        ),
    }


def run_fii_dii_flow_quantum_backtest(
    price_df: pd.DataFrame,
    net_positioning_wide: pd.DataFrame,
    cost_calc: CostCalculator,
    initial_capital: float,
    hold_days: int = 5,
    max_concurrent_positions: int = 5,
    quantile_threshold: float = 0.8,
    n_folds: int = 3,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
    n_qubits: int = 4,
    evolver_generations: int = 15,
    evolver_population: int = 30,
    evolver_top_k: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    price_df: 'date' + 'close' (see fii_dii_flow.py on instrument
    choice). net_positioning_wide: compute_net_positioning()'s full
    wide output (all client_type x instrument columns) - NOT just the
    single dii_net_index_future Series the rule-based backtest takes.

    Returns n_days_with_data, n_trades, n_folds, oos (this walk-forward
    run's pooled OOS trade report - every trade here is genuinely out-
    of-sample, no further train/oos split needed), fold_diagnostics
    (per-fold evolved genes, decision threshold, train accuracy - for
    inspecting what each fold actually learned, not just the pooled
    headline number), benchmark (the rule-based signal recomputed over
    this same eligible window), and comparison
    (compare_to_rule_based_benchmark's verdict).
    """
    price_df = price_df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    dates = price_df["date"].tolist()
    closes = price_df["close"].tolist()
    n = len(dates)

    raw_panel = build_institutional_flow_feature_panel(net_positioning_wide, lookbacks=lookbacks)
    if raw_panel.empty:
        raise RuntimeError("net_positioning_wide is empty - no institutional-flow features to build.")
    raw_panel = raw_panel.reindex(dates)
    raw_feature_cols = list(raw_panel.columns)

    label = _forward_return_label(closes, hold_days)

    df = raw_panel.reset_index(drop=True).copy()
    df["future_return"] = label
    df["date"] = dates

    min_train_days = max(MIN_TRAIN_DAYS, max(lookbacks) + 30)
    eligible_start = min_train_days
    n_eligible = n - eligible_start
    if n_eligible < n_folds * 20:
        raise RuntimeError(
            f"Only {n_eligible} eligible days after the {min_train_days}-day warmup; "
            f"need at least {n_folds * 20} for {n_folds} folds."
        )

    fold_ranges = _fold_boundaries(n_eligible, n_folds)

    entry_ok = pd.Series(False, index=range(n))
    fold_diagnostics: List[Dict[str, Any]] = []

    for fold_i, (fold_start_rel, fold_end_rel) in enumerate(fold_ranges):
        fold_start = eligible_start + fold_start_rel
        fold_end = eligible_start + fold_end_rel  # exclusive

        train_df = df.iloc[:fold_start].dropna(subset=["future_return"]).copy()
        if len(train_df) < min_train_days // 2:
            continue  # not enough labeled history yet - skip this fold's signals entirely

        evolver = FeatureEvolver(
            population_size=evolver_population,
            n_generations=evolver_generations,
            top_k=evolver_top_k,
            random_state=random_state + fold_i,
        )
        winners = evolver.evolve(train_df, raw_feature_cols, label_col="future_return")
        genes = [gene for gene, _fitness in winners]

        train_evolved = FeatureEvolver.apply_genes(train_df, genes)
        fold_slice = df.iloc[fold_start:fold_end].copy()
        fold_evolved = FeatureEvolver.apply_genes(fold_slice, genes)

        model_cols = raw_feature_cols + [gene.name() for gene in genes]
        X_train = train_evolved[model_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        y_train = (train_evolved["future_return"].to_numpy() > 0).astype(int)
        X_fold = fold_evolved[model_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()

        if len(np.unique(y_train)) < 2:
            continue  # degenerate label (all-up or all-down in this train slice) - can't fit a 2-class model

        clf = QuantumKernelClassifier(n_qubits=n_qubits, use_pca=True, pca_components=n_qubits)
        clf.fit(X_train, y_train)

        # Decision threshold from the TRAIN slice's OWN predicted
        # probabilities (top quantile_threshold quantile) - a
        # calibration choice made entirely from information available
        # before the fold starts, mirroring the rule-based signal's
        # "top quintile" selectivity without peeking at the fold.
        #
        # Scored on a bounded subsample of the train slice, not the
        # full thing: predict_proba's cost is a quantum-kernel
        # evaluation against every fitted support vector (up to the
        # classifier's own internal fit-time cap), so scoring it
        # against an uncapped, growing-every-fold train slice would
        # make later folds' calibration step far more expensive than
        # the fit itself - the actual bottleneck this comment replaces
        # (a single fold's calibration call alone took minutes at
        # ~250 train rows before this cap was added). A calibration
        # quantile is a distributional question - a bounded random
        # sample of the train slice answers it just as well.
        calib_size = min(len(X_train), 150)
        if len(X_train) > calib_size:
            calib_idx = np.random.default_rng(random_state + fold_i).choice(len(X_train), calib_size, replace=False)
            X_calib = X_train[calib_idx]
        else:
            X_calib = X_train
        train_proba_up = clf.predict_proba(X_calib)[:, 1]
        threshold = float(np.quantile(train_proba_up, quantile_threshold))

        fold_proba_up = clf.predict_proba(X_fold)[:, 1]
        fold_entries = fold_proba_up >= threshold
        entry_ok.iloc[fold_start:fold_end] = fold_entries

        fold_diagnostics.append({
            "fold": fold_i,
            "train_days": len(train_df),
            "fold_start_date": str(dates[fold_start]),
            "fold_end_date": str(dates[fold_end - 1]) if fold_end > fold_start else str(dates[fold_start]),
            "n_evolved_genes": len(genes),
            "evolved_gene_names": [gene.name() for gene in genes],
            "decision_threshold": threshold,
            "is_quantum": clf.is_quantum,
            "train_accuracy": clf.training_metrics.get("train_accuracy"),
            "n_signals_in_fold": int(fold_entries.sum()),
            "fold_n_days": fold_end - fold_start,
        })

    position_notional = initial_capital / max_concurrent_positions
    trades = simulate_concurrent_tranche_trades(
        dates, closes, entry_ok, hold_days, max_concurrent_positions, cost_calc, position_notional,
    )

    if not trades:
        return {
            "n_days_with_data": n, "n_trades": 0, "n_folds": len(fold_ranges),
            "oos": {}, "fold_diagnostics": fold_diagnostics,
        }

    trades_df = pd.DataFrame(trades).sort_values("exit_date").reset_index(drop=True)
    oos_report = _report_from_trades(trades_df, initial_capital)

    benchmark = _rule_based_benchmark_same_window(
        price_df, net_positioning_wide, cost_calc, initial_capital, eligible_start,
        flow_lookback_days=5, trailing_window=252, quantile_threshold=quantile_threshold,
        hold_days=hold_days, max_concurrent_positions=max_concurrent_positions,
    )
    comparison = compare_to_rule_based_benchmark(oos_report, benchmark)

    return {
        "n_days_with_data": n,
        "n_trades": len(trades_df),
        "n_folds": len(fold_ranges),
        "oos": oos_report,
        "fold_diagnostics": fold_diagnostics,
        "benchmark": {k: v for k, v in benchmark.items() if k not in ("period_returns", "period_dates")},
        "comparison": comparison,
    }
