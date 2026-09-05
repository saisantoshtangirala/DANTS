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
