import numpy as np
import pandas as pd
import pytest

from src.training.feature_evolution import ALL_OPS, UNARY_OPS, FeatureEvolver, Gene, permutation_test_ic


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    feature_a = rng.normal(size=n)
    feature_b = rng.normal(size=n)
    noise = rng.normal(scale=0.01, size=n)
    return pd.DataFrame(
        {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "noise_feature": rng.normal(size=n),
            # future_return strongly driven by feature_a, weakly by anything else
            "future_return": feature_a * 2.0 + noise,
        }
    )


def test_gene_name_unary_vs_binary():
    unary = Gene(feature_a="close_to_sma_20", feature_b=None, op="rolling_mean_5")
    binary = Gene(feature_a="rsi", feature_b="macd", op="ratio")

    assert unary.name() == "evolved_rolling_mean_5_close_to_sma_20"
    assert binary.name() == "evolved_rsi_ratio_macd"


def test_gene_to_dict_from_dict_roundtrip():
    gene = Gene(feature_a="a", feature_b="b", op="diff")
    restored = Gene.from_dict(gene.to_dict())
    assert restored == gene


def test_gene_evaluate_missing_feature_returns_none(synthetic_df):
    gene = Gene(feature_a="does_not_exist", feature_b="feature_b", op="ratio")
    assert gene.evaluate(synthetic_df) is None


def test_gene_evaluate_binary_missing_feature_b_returns_none(synthetic_df):
    gene = Gene(feature_a="feature_a", feature_b="does_not_exist", op="ratio")
    assert gene.evaluate(synthetic_df) is None


def test_gene_evaluate_ratio_matches_manual_computation(synthetic_df):
    gene = Gene(feature_a="feature_a", feature_b="feature_b", op="ratio")
    result = gene.evaluate(synthetic_df)
    expected = synthetic_df["feature_a"] / synthetic_df["feature_b"].replace(0, np.nan)
    pd.testing.assert_series_equal(result, expected)


def test_gene_evaluate_unary_rolling_mean(synthetic_df):
    gene = Gene(feature_a="feature_a", feature_b=None, op="rolling_mean_5")
    result = gene.evaluate(synthetic_df)
    expected = synthetic_df["feature_a"].rolling(window=5, min_periods=1).mean()
    pd.testing.assert_series_equal(result, expected)


def test_apply_genes_appends_named_columns_and_fills_failures(synthetic_df):
    good_gene = Gene(feature_a="feature_a", feature_b="feature_b", op="diff")
    bad_gene = Gene(feature_a="missing_col", feature_b=None, op="rolling_std_5")

    result = FeatureEvolver.apply_genes(synthetic_df, [good_gene, bad_gene])

    assert good_gene.name() in result.columns
    assert bad_gene.name() in result.columns
    assert (result[bad_gene.name()] == 0.0).all()
    assert np.isfinite(result[good_gene.name()]).all()


def test_evolve_ranks_informative_gene_above_noise(synthetic_df):
    evolver = FeatureEvolver(population_size=20, n_generations=8, top_k=3, random_state=1)
    feature_cols = ["feature_a", "feature_b", "noise_feature"]

    ranked = evolver.evolve(synthetic_df, feature_cols, label_col="future_return")

    assert len(ranked) > 0
    best_gene, best_fitness = ranked[0]
    assert best_fitness > 0
    # The best gene found should involve feature_a, since it drives future_return.
    assert best_gene.feature_a == "feature_a" or best_gene.feature_b == "feature_a"


def test_evolve_returns_empty_when_label_col_missing(synthetic_df):
    evolver = FeatureEvolver(population_size=5, n_generations=2)
    ranked = evolver.evolve(synthetic_df, ["feature_a"], label_col="does_not_exist")
    assert ranked == []


def test_evolve_returns_empty_when_no_feature_cols(synthetic_df):
    evolver = FeatureEvolver(population_size=5, n_generations=2)
    ranked = evolver.evolve(synthetic_df, [], label_col="future_return")
    assert ranked == []


def test_all_ops_contains_unary_ops():
    assert set(UNARY_OPS).issubset(set(ALL_OPS))


class TestPermutationTestIc:
    def test_pure_noise_gives_high_p_value(self):
        rng = np.random.default_rng(3)
        n = 300
        derived = pd.Series(rng.normal(size=n))
        label = rng.normal(size=n)  # unrelated to derived
        p = permutation_test_ic(derived, label, n_permutations=200, random_state=1)
        assert p > 0.05

    def test_strong_real_signal_gives_low_p_value(self):
        rng = np.random.default_rng(4)
        n = 300
        derived = pd.Series(rng.normal(size=n))
        label = derived.to_numpy() * 3.0 + rng.normal(scale=0.05, size=n)  # strongly driven by derived
        p = permutation_test_ic(derived, label, n_permutations=200, random_state=1)
        assert p < 0.01

    def test_zero_fitness_short_circuits_to_p_one(self):
        derived = pd.Series([1.0] * 20)  # constant -> zero IC
        label = np.arange(20, dtype=float)
        assert permutation_test_ic(derived, label) == 1.0

    def test_block_size_one_matches_default_iid_behavior(self):
        """block_size=1 must be indistinguishable from the plain i.i.d.
        shuffle this function always did before block_size existed - a
        default-preserving parameter, not a behavior change for existing
        callers."""
        rng = np.random.default_rng(7)
        n = 300
        derived = pd.Series(rng.normal(size=n))
        label = derived.to_numpy() * 2.0 + rng.normal(scale=0.1, size=n)
        p_default = permutation_test_ic(derived, label, n_permutations=200, random_state=1)
        p_explicit_block_1 = permutation_test_ic(derived, label, n_permutations=200, random_state=1, block_size=1)
        assert p_default == p_explicit_block_1

    def test_block_permutation_still_detects_strong_planted_signal(self):
        """block_size > 1 must not be so conservative it kills detection
        power entirely - a strong real signal should still score a low
        p-value."""
        rng = np.random.default_rng(8)
        n = 300
        derived = pd.Series(rng.normal(size=n))
        label = derived.to_numpy() * 3.0 + rng.normal(scale=0.05, size=n)
        p = permutation_test_ic(derived, label, n_permutations=200, random_state=1, block_size=5)
        assert p < 0.05

    def test_block_permutation_reduces_false_positive_rate_on_autocorrelated_series(self):
        """The actual bug this parameter fixes: an autocorrelated feature
        (cumulative-sum random walk, like net-positioning diffs) tested
        against an autocorrelated label (overlapping-window forward
        returns) with ZERO true relationship should trigger far fewer
        false "significant" results under block permutation than under
        the plain i.i.d. shuffle - matching the investigation's measured
        11.0% (i.i.d.) vs a materially lower rate under block_size=5,
        at the same Bonferroni-corrected alpha this codebase actually
        uses (0.05 / 20 candidates)."""
        rng = np.random.default_rng(0)
        n_trials = 50
        n_days = 250
        hold_days = 5
        alpha_corrected = 0.05 / 20

        def false_positive_count(block_size: int) -> int:
            local_rng = np.random.default_rng(0)
            count = 0
            for trial in range(n_trials):
                raw_feature = np.cumsum(local_rng.normal(0, 1, n_days + 5))
                feature = raw_feature[5:] - raw_feature[:-5]
                price = np.cumsum(local_rng.normal(0, 1, n_days + hold_days + 1))
                label = np.full(n_days, np.nan)
                for t in range(n_days - hold_days - 1):
                    label[t] = price[t + 1 + hold_days] - price[t + 1]
                valid = ~np.isnan(label)
                p = permutation_test_ic(
                    pd.Series(feature[valid]), label[valid], n_permutations=1000,
                    random_state=trial, block_size=block_size,
                )
                if p <= alpha_corrected:
                    count += 1
            return count

        iid_false_positives = false_positive_count(block_size=1)
        block_false_positives = false_positive_count(block_size=hold_days)

        # The i.i.d. case should reproduce the investigation's inflated
        # rate (materially above the ~0.25% nominal alpha at this sample
        # size); block permutation should score meaningfully fewer false
        # positives on the identical trials.
        assert iid_false_positives >= 3  # reproduces the known leak (not just noise)
        assert block_false_positives < iid_false_positives


class TestEvolveWithValidation:
    def test_rejects_gene_that_only_fits_training_noise(self):
        """A gene search over PURE NOISE features/label should find
        nothing that survives validation - if it does, the double-dip
        this method exists to fix is still present."""
        rng = np.random.default_rng(5)
        n = 150
        train_df = pd.DataFrame({
            "feature_a": rng.normal(size=n),
            "feature_b": rng.normal(size=n),
            "future_return": rng.normal(size=n),  # unrelated to features
        })
        val_df = pd.DataFrame({
            "feature_a": rng.normal(size=n),
            "feature_b": rng.normal(size=n),
            "future_return": rng.normal(size=n),
        })
        evolver = FeatureEvolver(population_size=30, n_generations=10, top_k=5, random_state=2)
        survivors = evolver.evolve_with_validation(
            train_df, val_df, ["feature_a", "feature_b"], n_candidates=15, n_permutations=100,
        )
        assert survivors == []

    def test_accepts_gene_that_is_genuinely_informative_in_both_slices(self):
        rng = np.random.default_rng(6)
        n = 200

        def make_slice(seed):
            r = np.random.default_rng(seed)
            feature_a = r.normal(size=n)
            noise_feature = r.normal(size=n)
            future_return = feature_a * 2.5 + r.normal(scale=0.1, size=n)
            return pd.DataFrame({"feature_a": feature_a, "noise_feature": noise_feature, "future_return": future_return})

        train_df = make_slice(10)
        val_df = make_slice(11)  # different draw, SAME underlying relationship

        evolver = FeatureEvolver(population_size=30, n_generations=10, top_k=3, random_state=3)
        survivors = evolver.evolve_with_validation(
            train_df, val_df, ["feature_a", "noise_feature"], n_candidates=10, n_permutations=1000,
        )
        assert len(survivors) > 0
        best_gene, val_ic, p_value = survivors[0]
        assert best_gene.feature_a == "feature_a" or best_gene.feature_b == "feature_a"
        assert val_ic > 0
        assert p_value <= 0.05

    def test_returns_empty_when_label_col_missing_from_either_slice(self):
        df = pd.DataFrame({"feature_a": [1.0, 2.0, 3.0] * 10})
        evolver = FeatureEvolver(population_size=5, n_generations=2)
        assert evolver.evolve_with_validation(df, df, ["feature_a"], label_col="missing") == []

    def test_embargo_rows_truncates_train_before_ga_search(self):
        """embargo_rows must drop rows off the END of df_train before the
        GA search sees it (not just before scoring survivors) - verified
        by spying on evolve()'s input length directly, not just the
        final output, since a wrong embargo that still lets the search
        overfit those rows wouldn't show up in the survivor list alone."""
        df_train = pd.DataFrame({
            "feature_a": np.arange(100, dtype=float),
            "future_return": np.arange(100, dtype=float),
        })
        val_df = pd.DataFrame({
            "feature_a": np.arange(50, dtype=float),
            "future_return": np.arange(50, dtype=float),
        })
        evolver = FeatureEvolver(population_size=5, n_generations=2, top_k=2, random_state=1)
        captured_lengths = []
        original_evolve = evolver.evolve

        def spy_evolve(df, feature_cols, label_col="future_return"):
            captured_lengths.append(len(df))
            return original_evolve(df, feature_cols, label_col=label_col)

        evolver.evolve = spy_evolve
        evolver.evolve_with_validation(
            df_train, val_df, ["feature_a"], n_candidates=5, n_permutations=50, embargo_rows=10,
        )
        assert captured_lengths == [90]

    def test_block_size_and_embargo_still_accept_a_genuine_signal(self):
        """Passing block_size and embargo_rows shouldn't make the method
        so conservative it rejects a real, strong, both-slices-genuine
        relationship - it should still find it, just with the corrected
        (autocorrelation-aware) significance test."""
        rng = np.random.default_rng(6)
        n = 200

        def make_slice(seed):
            r = np.random.default_rng(seed)
            feature_a = r.normal(size=n)
            noise_feature = r.normal(size=n)
            future_return = feature_a * 2.5 + r.normal(scale=0.1, size=n)
            return pd.DataFrame({"feature_a": feature_a, "noise_feature": noise_feature, "future_return": future_return})

        train_df = make_slice(10)
        val_df = make_slice(11)

        evolver = FeatureEvolver(population_size=30, n_generations=10, top_k=3, random_state=3)
        survivors = evolver.evolve_with_validation(
            train_df, val_df, ["feature_a", "noise_feature"], n_candidates=10, n_permutations=1000,
            block_size=5, embargo_rows=5,
        )
        assert len(survivors) > 0
        best_gene, val_ic, p_value = survivors[0]
        assert best_gene.feature_a == "feature_a" or best_gene.feature_b == "feature_a"
        assert p_value <= 0.05
