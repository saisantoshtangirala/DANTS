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
        Fold i: train on everything strictly BEFORE the fold, itself
                split chronologically into inner_train / inner_val
                -> evolve genes on inner_train, VALIDATE them on
                   inner_val - a permutation-tested, Bonferroni-
                   corrected significance bar
                   (FeatureEvolver.evolve_with_validation), keeping
                   only genes that prove themselves on data the search
                   never touched (often none - see below on why this
                   exists)
                -> refit the surviving genes' columns on the FULL
                   train slice (inner_train + inner_val recombined -
                   inner_val's job was gating features, not shrinking
                   the final fit)
                -> fit `classifier_factory()` on that
                -> decision threshold = top `quantile_threshold`
                   quantile of the FITTED MODEL's OWN predicted
                   probabilities on its train slice (a calibration
                   choice made entirely from information available
                   before the fold starts - not tuned on the fold's
                   own outcome)
                -> apply that fixed model + threshold to the fold's
                   dates only, producing this fold's entry signals

Why the nested inner_train/inner_val split exists: the first version of
this module scored evolved genes' fitness on the SAME train slice a
classifier was then fit on - a double-dip (nothing had to prove itself
on data the search hadn't already seen), with zero correction for how
many combinations (population_size x n_generations, per fold) were
tried. Run for real on 5 years of NSE data, it produced textbook
overfitting: 85-89% per-fold train accuracy, but a pooled OOS Sharpe of
0.13 (p=0.80 - statistically indistinguishable from noise) against the
rule-based benchmark's Sharpe 1.19 (p=0.02) over the identical window.
evolve_with_validation (src/training/feature_evolution.py) fixes the
double-dip directly; the classifier is pluggable via classifier_factory
specifically so a cheap, heavily-regularized classical model can be
tried FIRST (run_fii_dii_flow_classical_ml_backtest, L1 logistic
regression) - far more sample-efficient than a quantum kernel fit
capped at 150 rows - before spending RunPod GPU budget re-testing
QuantumKernelClassifier (run_fii_dii_flow_quantum_backtest) on the same
disciplined search.

Even with that fix, the FIRST classical run (nested validation, no
block permutation yet) still lost to the benchmark - and a deeper dive
found the nested-validation permutation test itself was still leaky:
net-positioning features are cumulative sums (random walks) and
hold_days-overlapping forward-return labels are autocorrelated, and a
plain i.i.d. label shuffle destroys that structure in the null
distribution while leaving it intact in the real (unshuffled) score. A
synthetic test with ZERO true relationship between such a feature and
label found i.i.d. shuffling falsely "significant" at a Bonferroni-
corrected alpha=0.0025 in 22/200 trials (11.0%) vs. an i.i.d.-feature
control's 1/200 (0.5%) - a ~20x inflation. Three real-data ablations
confirmed the consequence: removing genetic evolution (raw features
only, same leaky test) improved OOS Sharpe from -0.16 to +0.14 -
evolution was making things WORSE, not neutral - while restricting to
ONLY the one already-validated feature (dii_net_index_future_diff5)
gave Sharpe 1.04 at p=0.043, nearly matching the benchmark's own
1.12/p=0.030. That combination (a plain classifier can trade the known
feature almost as well as the hand-tuned rule, but the wider 48-feature
search finds nothing but harm) points at too little independent data
for that search, not insufficient model capacity - see
evolver_block_size/embargo_rows below for the leaky-test fix, and
run_fii_dii_flow_pooled_ml_backtest for the cross-sectional pooling
built to attack the actual data-volume bottleneck.

Every fold's entry signals get combined into one full-history
admissibility series and run through
fii_dii_flow.simulate_concurrent_tranche_trades - the EXACT SAME
execution engine (same 1-day publication lag, same hold_days exit,
same CostCalculator) the rule-based backtest uses, so every model
tested here is apples-to-apples with the rule and with each other: they
differ only in which days pass the entry test, nothing about how a
passing day becomes a trade.

Adoption gate (compare_to_rule_based_benchmark): this module does NOT
decide for itself that a model is better. A run is only recommended for
paper trading if its own walk-forward OOS Sharpe ratio AND one-sample
significance both beat the rule-based benchmark - recomputed fresh over
the SAME eligible date range this run's folds cover (never a stale
hardcoded number from a previous run), per this project's standing rule
against ever fabricating or spinning a backtest result. If it doesn't
clear that bar, the documented fallback is periodic walk-forward
re-calibration of the existing rule's own parameters, not replacing a
validated simple rule with an unproven, more complex one. Either way,
this module never touches src/trading/fii_dii_flow_paper.py directly -
swapping paper trading over to a model this finds would be a separate,
explicit step taken only after a run here recommends it.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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


def compare_to_rule_based_benchmark(model_oos: Dict[str, Any], benchmark: Dict[str, Any]) -> Dict[str, Any]:
    """Explicit adoption gate. Whichever model produced `model_oos`
    (quantum kernel, classical logistic regression, or anything else
    run_fii_dii_flow_ml_backtest is pointed at) only counts as having
    beaten the validated rule-based signal if its walk-forward OOS
    Sharpe is higher AND its one-sample significance is at least as
    strong - a partial win (e.g. higher Sharpe but a weaker p-value) is
    reported as "does not beat benchmark", not spun as a win. See this
    module's docstring for the fallback if it doesn't clear this bar."""
    model_returns = model_oos.get("period_returns", [])
    model_sig = one_sample_significance_test(model_returns)
    model_sharpe = model_oos.get("sharpe_ratio", 0.0) or 0.0
    model_p = model_sig.get("p_value")

    benchmark_returns = benchmark.get("period_returns", [])
    benchmark_sig = one_sample_significance_test(benchmark_returns)
    benchmark_sharpe = benchmark.get("sharpe_ratio", 0.0) or 0.0
    benchmark_p = benchmark_sig.get("p_value")

    beats_sharpe = model_sharpe > benchmark_sharpe
    beats_significance = (
        model_p is not None and benchmark_p is not None and model_p <= benchmark_p
    )
    verdict = "beats_benchmark" if (beats_sharpe and beats_significance) else "does_not_beat_benchmark"

    return {
        "model_oos_sharpe": model_sharpe,
        "model_oos_p_value": model_p,
        "model_oos_n_trades": model_sig.get("n_trades", 0),
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


def run_fii_dii_flow_ml_backtest(
    price_df: pd.DataFrame,
    net_positioning_wide: pd.DataFrame,
    cost_calc: CostCalculator,
    initial_capital: float,
    classifier_factory: Callable[[], Any],
    model_label: str,
    hold_days: int = 5,
    max_concurrent_positions: int = 5,
    quantile_threshold: float = 0.8,
    n_folds: int = 3,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
    inner_val_frac: float = 0.25,
    n_permutations: int = 1000,
    permutation_alpha: float = 0.05,
    evolver_generations: int = 15,
    evolver_population: int = 30,
    evolver_top_k: int = 5,
    evolver_n_candidates: int = 20,
    evolver_block_size: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    price_df: 'date' + 'close' (see fii_dii_flow.py on instrument
    choice). net_positioning_wide: compute_net_positioning()'s full
    wide output (all client_type x instrument columns) - NOT just the
    single dii_net_index_future Series the rule-based backtest takes.

    classifier_factory: called fresh for each fold, must return an
    object with .fit(X, y) and .predict_proba(X) (2 columns, [:, 1] =
    P(up)) - e.g. QuantumKernelClassifier or sklearn's
    LogisticRegression. model_label: a short string identifying which
    model this run used (e.g. "quantum_kernel",
    "logistic_regression_l1"), carried through to the result for
    logging/Telegram clarity when comparing model types.

    evolver_block_size: forwarded to FeatureEvolver.evolve_with_validation
    as block_size (default None -> hold_days) - shuffles contiguous
    blocks of the label during the inner-validation permutation test
    instead of individual points, and embargo_rows=hold_days drops the
    last hold_days rows of each fold's inner_train before scoring gene
    fitness on it. Both address the same root cause: net-positioning
    features and hold_days-overlapping labels are autocorrelated, and a
    plain i.i.d. shuffle was measured to falsely pass genes ~20x more
    often than the nominal alpha in that situation - see this module's
    docstring and permutation_test_ic's for the measured numbers.

    Returns model_label, n_days_with_data, n_trades, n_folds, oos (this
    walk-forward run's pooled OOS trade report - every trade here is
    genuinely out-of-sample, no further train/oos split needed),
    fold_diagnostics (per-fold evolved genes that survived validation,
    their permutation p-values, inner_train/inner_val day counts,
    decision threshold, train accuracy - for inspecting what each fold
    actually learned, not just the pooled headline number),
    cross_fold_consistency (which raw features recur among survivors
    across >=2 folds - a real signal should show up more than once),
    benchmark (the rule-based signal recomputed over this same eligible
    window), and comparison (compare_to_rule_based_benchmark's verdict).
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

    block_size = evolver_block_size if evolver_block_size is not None else hold_days

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
    survivor_raw_features_by_fold: List[set] = []

    for fold_i, (fold_start_rel, fold_end_rel) in enumerate(fold_ranges):
        fold_start = eligible_start + fold_start_rel
        fold_end = eligible_start + fold_end_rel  # exclusive

        train_df = df.iloc[:fold_start].dropna(subset=["future_return"]).copy()
        if len(train_df) < min_train_days // 2:
            continue  # not enough labeled history yet - skip this fold's signals entirely

        # Chronological inner split: the last inner_val_frac of the train
        # slice is held out from the genetic search entirely, used only
        # to validate which evolved features (if any) are trusted -
        # fixes the double-dip a single-slice search has. See
        # FeatureEvolver.evolve_with_validation's docstring.
        inner_split = int(len(train_df) * (1 - inner_val_frac))
        inner_train_df = train_df.iloc[:inner_split]
        inner_val_df = train_df.iloc[inner_split:]

        survivors: List[Tuple[Any, float, float]] = []
        if len(inner_train_df) >= 30 and len(inner_val_df) >= 30:
            evolver = FeatureEvolver(
                population_size=evolver_population,
                n_generations=evolver_generations,
                top_k=evolver_top_k,
                random_state=random_state + fold_i,
            )
            survivors = evolver.evolve_with_validation(
                inner_train_df, inner_val_df, raw_feature_cols, label_col="future_return",
                n_candidates=evolver_n_candidates, n_permutations=n_permutations, alpha=permutation_alpha,
                block_size=block_size, embargo_rows=hold_days,
            )
        genes = [gene for gene, _val_ic, _p in survivors]
        survivor_raw_features_by_fold.append(
            {gene.feature_a for gene in genes} | {gene.feature_b for gene in genes if gene.feature_b}
        )

        # Refit the surviving genes' columns on the FULL train slice
        # (inner_train + inner_val recombined) - inner_val's only job
        # was gating which features are trusted, not shrinking the
        # final model's training data.
        train_evolved = FeatureEvolver.apply_genes(train_df, genes)
        fold_slice = df.iloc[fold_start:fold_end].copy()
        fold_evolved = FeatureEvolver.apply_genes(fold_slice, genes)

        model_cols = raw_feature_cols + [gene.name() for gene in genes]
        X_train = train_evolved[model_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        y_train = (train_evolved["future_return"].to_numpy() > 0).astype(int)
        X_fold = fold_evolved[model_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()

        if len(np.unique(y_train)) < 2:
            fold_diagnostics.append({
                "fold": fold_i, "train_days": len(train_df), "skipped_reason": "degenerate_label",
            })
            continue  # all-up or all-down train label - can't fit a 2-class model

        clf = classifier_factory()
        clf.fit(X_train, y_train)

        # Decision threshold from the TRAIN slice's OWN predicted
        # probabilities (top quantile_threshold quantile) - a
        # calibration choice made entirely from information available
        # before the fold starts, mirroring the rule-based signal's
        # "top quintile" selectivity without peeking at the fold.
        #
        # Scored on a bounded subsample of the train slice, not the
        # full thing: some classifiers (QuantumKernelClassifier)
        # charge a real cost per predict_proba call proportional to
        # the number of fitted support vectors, so scoring against an
        # uncapped, growing-every-fold train slice would make later
        # folds' calibration step far more expensive than the fit
        # itself - the actual bottleneck this comment replaces (a
        # single fold's calibration call alone took minutes at ~250
        # train rows before this cap was added). A calibration
        # quantile is a distributional question - a bounded random
        # sample of the train slice answers it just as well, for any
        # classifier. The same bounded sample doubles as the
        # train_accuracy diagnostic below, for the same reason.
        calib_size = min(len(X_train), 150)
        if len(X_train) > calib_size:
            calib_idx = np.random.default_rng(random_state + fold_i).choice(len(X_train), calib_size, replace=False)
            X_calib = X_train[calib_idx]
            y_calib = y_train[calib_idx]
        else:
            X_calib = X_train
            y_calib = y_train
        train_proba_up = clf.predict_proba(X_calib)[:, 1]
        threshold = float(np.quantile(train_proba_up, quantile_threshold))

        # Via predict_proba's [:, 1] = P(up) convention, not .predict() -
        # different classifier implementations use different label
        # conventions for predict() (QuantumKernelClassifier returns
        # {-1, 1}, sklearn returns whatever classes .fit() saw, here
        # {0, 1}), so comparing .predict()'s output directly against
        # y_train would silently misscore any classifier that doesn't
        # happen to share that exact convention.
        train_pred_up = (train_proba_up >= 0.5).astype(int)
        train_accuracy = float(np.mean(train_pred_up == y_calib))

        fold_proba_up = clf.predict_proba(X_fold)[:, 1]
        fold_entries = fold_proba_up >= threshold
        entry_ok.iloc[fold_start:fold_end] = fold_entries

        fold_diagnostics.append({
            "fold": fold_i,
            "train_days": len(train_df),
            "inner_train_days": len(inner_train_df),
            "inner_val_days": len(inner_val_df),
            "fold_start_date": str(dates[fold_start]),
            "fold_end_date": str(dates[fold_end - 1]) if fold_end > fold_start else str(dates[fold_start]),
            "n_genes_evaluated": evolver_n_candidates,
            "n_genes_passed_validation": len(genes),
            "evolved_gene_names": [gene.name() for gene in genes],
            "permutation_p_values": [round(p, 5) for _g, _v, p in survivors],
            "decision_threshold": threshold,
            "is_quantum": getattr(clf, "is_quantum", None),
            "train_accuracy": train_accuracy,
            "n_signals_in_fold": int(fold_entries.sum()),
            "fold_n_days": fold_end - fold_start,
        })

    # Cross-fold consistency: a real signal should recur, not appear once
    # and vanish - the same bar the original hand-found DII feature had
    # to clear ("significant in both halves"). Surfaced honestly, not
    # used to silently include/exclude anything from the backtest itself.
    feature_fold_counts: Dict[str, int] = {}
    for feats in survivor_raw_features_by_fold:
        for feat in feats:
            feature_fold_counts[feat] = feature_fold_counts.get(feat, 0) + 1
    cross_fold_consistency = {
        "recurring_raw_features": sorted(f for f, c in feature_fold_counts.items() if c >= 2),
        "feature_fold_counts": feature_fold_counts,
    }

    position_notional = initial_capital / max_concurrent_positions
    trades = simulate_concurrent_tranche_trades(
        dates, closes, entry_ok, hold_days, max_concurrent_positions, cost_calc, position_notional,
    )

    if not trades:
        return {
            "model_label": model_label,
            "n_days_with_data": n, "n_trades": 0, "n_folds": len(fold_ranges),
            "oos": {}, "fold_diagnostics": fold_diagnostics,
            "cross_fold_consistency": cross_fold_consistency,
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
        "model_label": model_label,
        "n_days_with_data": n,
        "n_trades": len(trades_df),
        "n_folds": len(fold_ranges),
        "oos": oos_report,
        "fold_diagnostics": fold_diagnostics,
        "cross_fold_consistency": cross_fold_consistency,
        "benchmark": {k: v for k, v in benchmark.items() if k not in ("period_returns", "period_dates")},
        "comparison": comparison,
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
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """Backward-compatible wrapper around run_fii_dii_flow_ml_backtest -
    same signature this had before nested validation and the pluggable
    classifier were added, using QuantumKernelClassifier as the model.
    See that function's docstring for the full mechanism and this
    module's docstring for why this now uses a nested inner_train/
    inner_val split instead of the single-slice search this originally
    shipped with."""
    return run_fii_dii_flow_ml_backtest(
        price_df, net_positioning_wide, cost_calc, initial_capital,
        classifier_factory=lambda: QuantumKernelClassifier(
            n_qubits=n_qubits, use_pca=True, pca_components=n_qubits, use_gpu=use_gpu,
        ),
        model_label="quantum_kernel",
        hold_days=hold_days, max_concurrent_positions=max_concurrent_positions,
        quantile_threshold=quantile_threshold, n_folds=n_folds, lookbacks=lookbacks,
        evolver_generations=evolver_generations, evolver_population=evolver_population,
        evolver_top_k=evolver_top_k, random_state=random_state,
    )


def run_fii_dii_flow_classical_ml_backtest(
    price_df: pd.DataFrame,
    net_positioning_wide: pd.DataFrame,
    cost_calc: CostCalculator,
    initial_capital: float,
    hold_days: int = 5,
    max_concurrent_positions: int = 5,
    quantile_threshold: float = 0.8,
    n_folds: int = 3,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
    evolver_generations: int = 15,
    evolver_population: int = 30,
    evolver_top_k: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Same walk-forward, nested-validation, and benchmark-comparison
    machinery as run_fii_dii_flow_quantum_backtest, but with an
    L1-regularized logistic regression instead of QuantumKernelClassifier.
    L1 does implicit feature selection and is far more sample-efficient
    than a quantum kernel fit capped at 150 rows - a cheap way to test
    "is there any real, generalizing signal in the wider institutional-
    flow feature space at all" before spending RunPod GPU budget
    re-testing the quantum version on the same disciplined search. See
    this module's docstring for the intended rollout order (run this
    first; only re-run the quantum version if this one beats the
    benchmark)."""
    return run_fii_dii_flow_ml_backtest(
        price_df, net_positioning_wide, cost_calc, initial_capital,
        classifier_factory=lambda: LogisticRegression(
            l1_ratio=1, solver="liblinear", C=0.1, class_weight="balanced", random_state=random_state,
        ),
        model_label="logistic_regression_l1",
        hold_days=hold_days, max_concurrent_positions=max_concurrent_positions,
        quantile_threshold=quantile_threshold, n_folds=n_folds, lookbacks=lookbacks,
        evolver_generations=evolver_generations, evolver_population=evolver_population,
        evolver_top_k=evolver_top_k, random_state=random_state,
    )


def _first_index_on_or_after(dates: List[Any], target: Any) -> int:
    """First position in dates (assumed sorted) whose value is >=
    target, or len(dates) if none - used to translate a reference
    instrument's eligible_start date into another instrument's own
    positional index."""
    for i, d in enumerate(dates):
        if d >= target:
            return i
    return len(dates)


def _combine_period_reports(reports: List[Dict[str, Any]], initial_capital: float) -> Dict[str, Any]:
    """Pools several already-computed reports' period_returns/
    period_dates (each produced by _report_from_trades) into one
    aggregate report, using _report_from_trades itself on a
    reconstructed pnl_pct/exit_date frame - used to combine multiple
    instruments' individually-computed rule-based benchmark reports
    into one pooled Sharpe/significance figure, the same way pooled
    model trades are combined from a single concatenated trades_df."""
    returns: List[float] = []
    dates: List[Any] = []
    for r in reports:
        returns.extend(r.get("period_returns", []))
        dates.extend(r.get("period_dates", []))
    if not returns:
        return {"n_trades": 0}
    order = np.argsort(pd.to_datetime(pd.Series(dates)).to_numpy())
    pooled = pd.DataFrame({"pnl_pct": np.array(returns)[order], "exit_date": np.array(dates, dtype=object)[order]})
    return _report_from_trades(pooled, initial_capital)


def run_fii_dii_flow_pooled_ml_backtest(
    price_dfs: Dict[str, pd.DataFrame],
    net_positioning_wide: pd.DataFrame,
    cost_calc: CostCalculator,
    initial_capital: float,
    classifier_factory: Callable[[], Any],
    model_label: str,
    hold_days: int = 5,
    max_concurrent_positions: int = 5,
    quantile_threshold: float = 0.8,
    n_folds: int = 3,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
    inner_val_frac: float = 0.25,
    n_permutations: int = 1000,
    permutation_alpha: float = 0.05,
    evolver_generations: int = 15,
    evolver_population: int = 30,
    evolver_top_k: int = 5,
    evolver_n_candidates: int = 20,
    evolver_block_size: Optional[int] = None,
    random_state: int = 42,
    reference_symbol: Optional[str] = None,
) -> Dict[str, Any]:
    """Cross-sectional generalization of run_fii_dii_flow_ml_backtest:
    pools MULTIPLE instruments' own price series against the SAME
    shared institutional-flow feature panel (net_positioning_wide is one
    nationwide F&O-market aggregate, identical across every instrument -
    only each instrument's own forward-return label differs). This
    multiplies the number of independent (date, instrument) label draws
    available to the genetic search and classifier fit without adding
    any new raw features - built to attack the "not enough independent
    data" bottleneck this session's investigation found in the single-
    instrument search (see this module's docstring), not to widen the
    search space further.

    price_dfs: {symbol: price_df} - each price_df is 'date' + 'close'
    for one cash-tradable instrument (e.g. config.yaml's
    data.symbols.sector_etf_universe). Each instrument keeps its own
    trading calendar (an ETF listed partway through the window simply
    contributes fewer rows, not NaN-padded ones).

    Walk-forward folds are defined on `reference_symbol`'s own date
    calendar (default: whichever instrument in price_dfs has the
    longest history) - every other instrument's rows are pooled into
    each fold's train/OOS slice by DATE cutoff (`< fold_start_date` /
    `fold_start_date <= date < fold_end_date`), not by row position,
    since instruments can have different calendars/history lengths.

    Per fold: pool every instrument's own labeled rows before
    fold_start_date into ONE train_df, run evolve_with_validation and
    fit classifier_factory() ONCE on that pooled frame (this is where
    the sample-size win comes from), then apply the single fitted
    model + threshold separately to each instrument's own fold-period
    rows to generate that instrument's entry_ok, and run
    simulate_concurrent_tranche_trades per instrument (capital
    allocation and exits are inherently per-instrument - this is a
    diagnostic pooling multiple INDEPENDENT single-instrument backtests'
    trades for one combined OOS report, not a capital-constrained
    combined portfolio: each instrument's own simulate_concurrent_
    tranche_trades call independently sizes tranches off the full
    initial_capital, exactly like every other diagnostic in this
    module).

    Since the shared feature panel's values don't vary by instrument on
    a given date, the fitted model's predicted probability for a date
    is identical across every instrument still trading that date - a
    real property of this signal (nationwide institutional flow doesn't
    predict which stock/sector moves, only that "the market" broadly
    might), not a bug: a qualifying day can open a tranche in several
    pooled instruments simultaneously.

    Returns the same shape as run_fii_dii_flow_ml_backtest plus
    n_instruments/instruments; fold_diagnostics adds
    n_instruments_pooled. benchmark/comparison pool each instrument's
    own rule-based benchmark (recomputed over that instrument's slice of
    the identical eligible window) via _combine_period_reports, so the
    win/lose gate stays apples-to-apples with the pooled model result.
    """
    if not price_dfs:
        raise RuntimeError("price_dfs is empty - need at least one instrument to pool.")

    block_size = evolver_block_size if evolver_block_size is not None else hold_days

    raw_panel = build_institutional_flow_feature_panel(net_positioning_wide, lookbacks=lookbacks)
    if raw_panel.empty:
        raise RuntimeError("net_positioning_wide is empty - no institutional-flow features to build.")
    raw_feature_cols = list(raw_panel.columns)

    symbol_frames: Dict[str, pd.DataFrame] = {}
    symbol_dates: Dict[str, List[Any]] = {}
    symbol_closes: Dict[str, List[float]] = {}
    for symbol, p_df in price_dfs.items():
        p_df = p_df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        if p_df.empty:
            continue
        dates_s = p_df["date"].tolist()
        closes_s = p_df["close"].tolist()
        df_s = raw_panel.reindex(dates_s).reset_index(drop=True)
        df_s["future_return"] = _forward_return_label(closes_s, hold_days)
        df_s["date"] = dates_s
        symbol_frames[symbol] = df_s
        symbol_dates[symbol] = dates_s
        symbol_closes[symbol] = closes_s

    if not symbol_frames:
        raise RuntimeError("No usable price data (date+close) for any instrument in price_dfs.")

    reference_symbol = reference_symbol or max(symbol_dates, key=lambda s: len(symbol_dates[s]))
    ref_dates = symbol_dates[reference_symbol]
    n_ref = len(ref_dates)

    min_train_days = max(MIN_TRAIN_DAYS, max(lookbacks) + 30)
    eligible_start = min_train_days
    n_eligible = n_ref - eligible_start
    if n_eligible < n_folds * 20:
        raise RuntimeError(
            f"Only {n_eligible} eligible days (on reference instrument '{reference_symbol}') after "
            f"the {min_train_days}-day warmup; need at least {n_folds * 20} for {n_folds} folds."
        )

    fold_ranges = _fold_boundaries(n_eligible, n_folds)

    entry_ok_by_symbol: Dict[str, pd.Series] = {
        s: pd.Series(False, index=range(len(symbol_dates[s]))) for s in symbol_frames
    }
    fold_diagnostics: List[Dict[str, Any]] = []
    survivor_raw_features_by_fold: List[set] = []

    for fold_i, (fold_start_rel, fold_end_rel) in enumerate(fold_ranges):
        fold_start_idx = eligible_start + fold_start_rel
        fold_end_idx = eligible_start + fold_end_rel  # exclusive, on the reference calendar
        fold_start_date = ref_dates[fold_start_idx]
        fold_end_date_exclusive = ref_dates[fold_end_idx] if fold_end_idx < n_ref else None
        fold_end_date_inclusive = ref_dates[fold_end_idx - 1] if fold_end_idx > fold_start_idx else fold_start_date

        train_parts = [
            df_s[df_s["date"] < fold_start_date].dropna(subset=["future_return"])
            for df_s in symbol_frames.values()
        ]
        train_parts = [p for p in train_parts if not p.empty]
        if not train_parts:
            continue
        train_df = pd.concat(train_parts, ignore_index=True).sort_values("date").reset_index(drop=True)
        if len(train_df) < min_train_days // 2:
            continue

        inner_split = int(len(train_df) * (1 - inner_val_frac))
        inner_train_df = train_df.iloc[:inner_split]
        inner_val_df = train_df.iloc[inner_split:]

        survivors: List[Tuple[Any, float, float]] = []
        if len(inner_train_df) >= 30 and len(inner_val_df) >= 30:
            evolver = FeatureEvolver(
                population_size=evolver_population, n_generations=evolver_generations,
                top_k=evolver_top_k, random_state=random_state + fold_i,
            )
            survivors = evolver.evolve_with_validation(
                inner_train_df, inner_val_df, raw_feature_cols, label_col="future_return",
                n_candidates=evolver_n_candidates, n_permutations=n_permutations, alpha=permutation_alpha,
                block_size=block_size, embargo_rows=hold_days,
            )
        genes = [gene for gene, _val_ic, _p in survivors]
        survivor_raw_features_by_fold.append(
            {gene.feature_a for gene in genes} | {gene.feature_b for gene in genes if gene.feature_b}
        )

        train_evolved = FeatureEvolver.apply_genes(train_df, genes)
        model_cols = raw_feature_cols + [gene.name() for gene in genes]
        X_train = train_evolved[model_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        y_train = (train_evolved["future_return"].to_numpy() > 0).astype(int)

        if len(np.unique(y_train)) < 2:
            fold_diagnostics.append({
                "fold": fold_i, "train_days": len(train_df), "skipped_reason": "degenerate_label",
            })
            continue

        clf = classifier_factory()
        clf.fit(X_train, y_train)

        calib_size = min(len(X_train), 150)
        if len(X_train) > calib_size:
            calib_idx = np.random.default_rng(random_state + fold_i).choice(len(X_train), calib_size, replace=False)
            X_calib = X_train[calib_idx]
            y_calib = y_train[calib_idx]
        else:
            X_calib = X_train
            y_calib = y_train
        train_proba_up = clf.predict_proba(X_calib)[:, 1]
        threshold = float(np.quantile(train_proba_up, quantile_threshold))
        train_pred_up = (train_proba_up >= 0.5).astype(int)
        train_accuracy = float(np.mean(train_pred_up == y_calib))

        n_signals_this_fold = 0
        for s, df_s in symbol_frames.items():
            mask = df_s["date"] >= fold_start_date
            if fold_end_date_exclusive is not None:
                mask &= df_s["date"] < fold_end_date_exclusive
            fold_slice = df_s[mask]
            if fold_slice.empty:
                continue
            fold_evolved = FeatureEvolver.apply_genes(fold_slice, genes)
            X_fold = fold_evolved[model_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
            fold_proba_up = clf.predict_proba(X_fold)[:, 1]
            fold_entries = fold_proba_up >= threshold
            entry_ok_by_symbol[s].iloc[fold_slice.index.to_numpy()] = fold_entries
            n_signals_this_fold += int(fold_entries.sum())

        fold_diagnostics.append({
            "fold": fold_i,
            "train_days": len(train_df),
            "inner_train_days": len(inner_train_df),
            "inner_val_days": len(inner_val_df),
            "fold_start_date": str(fold_start_date),
            "fold_end_date": str(fold_end_date_inclusive),
            "n_instruments_pooled": len(symbol_frames),
            "n_genes_evaluated": evolver_n_candidates,
            "n_genes_passed_validation": len(genes),
            "evolved_gene_names": [gene.name() for gene in genes],
            "permutation_p_values": [round(p, 5) for _g, _v, p in survivors],
            "decision_threshold": threshold,
            "is_quantum": getattr(clf, "is_quantum", None),
            "train_accuracy": train_accuracy,
            "n_signals_in_fold": n_signals_this_fold,
        })

    feature_fold_counts: Dict[str, int] = {}
    for feats in survivor_raw_features_by_fold:
        for feat in feats:
            feature_fold_counts[feat] = feature_fold_counts.get(feat, 0) + 1
    cross_fold_consistency = {
        "recurring_raw_features": sorted(f for f, c in feature_fold_counts.items() if c >= 2),
        "feature_fold_counts": feature_fold_counts,
    }

    position_notional = initial_capital / max_concurrent_positions
    all_trades: List[Dict[str, Any]] = []
    benchmark_reports: List[Dict[str, Any]] = []
    eligible_start_date = ref_dates[eligible_start]
    for s, df_s in symbol_frames.items():
        dates_s = symbol_dates[s]
        closes_s = symbol_closes[s]
        trades_s = simulate_concurrent_tranche_trades(
            dates_s, closes_s, entry_ok_by_symbol[s], hold_days, max_concurrent_positions, cost_calc, position_notional,
        )
        all_trades.extend(trades_s)

        bench = _rule_based_benchmark_same_window(
            pd.DataFrame({"date": dates_s, "close": closes_s}), net_positioning_wide, cost_calc, initial_capital,
            eligible_start=_first_index_on_or_after(dates_s, eligible_start_date),
            flow_lookback_days=5, trailing_window=252, quantile_threshold=quantile_threshold,
            hold_days=hold_days, max_concurrent_positions=max_concurrent_positions,
        )
        if bench.get("n_trades", 0) > 0:
            benchmark_reports.append(bench)

    if not all_trades:
        return {
            "model_label": model_label,
            "n_instruments": len(symbol_frames), "instruments": sorted(symbol_frames),
            "n_trades": 0, "n_folds": len(fold_ranges),
            "oos": {}, "fold_diagnostics": fold_diagnostics,
            "cross_fold_consistency": cross_fold_consistency,
        }

    trades_df = pd.DataFrame(all_trades).sort_values("exit_date").reset_index(drop=True)
    oos_report = _report_from_trades(trades_df, initial_capital)
    benchmark = _combine_period_reports(benchmark_reports, initial_capital)
    comparison = compare_to_rule_based_benchmark(oos_report, benchmark)

    return {
        "model_label": model_label,
        "n_instruments": len(symbol_frames),
        "instruments": sorted(symbol_frames),
        "n_trades": len(trades_df),
        "n_folds": len(fold_ranges),
        "oos": oos_report,
        "fold_diagnostics": fold_diagnostics,
        "cross_fold_consistency": cross_fold_consistency,
        "benchmark": {k: v for k, v in benchmark.items() if k not in ("period_returns", "period_dates")},
        "comparison": comparison,
    }


def run_fii_dii_flow_pooled_classical_ml_backtest(
    price_dfs: Dict[str, pd.DataFrame],
    net_positioning_wide: pd.DataFrame,
    cost_calc: CostCalculator,
    initial_capital: float,
    hold_days: int = 5,
    max_concurrent_positions: int = 5,
    quantile_threshold: float = 0.8,
    n_folds: int = 3,
    lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
    evolver_generations: int = 15,
    evolver_population: int = 30,
    evolver_top_k: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Same pooled, cross-sectional, nested-validation machinery as
    run_fii_dii_flow_pooled_ml_backtest, with L1-regularized logistic
    regression - classical only, matching this module's established
    classical-first rollout discipline (no quantum/GPU spend on pooling
    until a classical result actually beats the benchmark)."""
    return run_fii_dii_flow_pooled_ml_backtest(
        price_dfs, net_positioning_wide, cost_calc, initial_capital,
        classifier_factory=lambda: LogisticRegression(
            l1_ratio=1, solver="liblinear", C=0.1, class_weight="balanced", random_state=random_state,
        ),
        model_label="pooled_logistic_regression_l1",
        hold_days=hold_days, max_concurrent_positions=max_concurrent_positions,
        quantile_threshold=quantile_threshold, n_folds=n_folds, lookbacks=lookbacks,
        evolver_generations=evolver_generations, evolver_population=evolver_population,
        evolver_top_k=evolver_top_k, random_state=random_state,
    )
