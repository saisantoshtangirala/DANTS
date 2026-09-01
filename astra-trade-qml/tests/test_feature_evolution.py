import numpy as np
import pandas as pd
import pytest

from src.training.feature_evolution import ALL_OPS, UNARY_OPS, FeatureEvolver, Gene


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
