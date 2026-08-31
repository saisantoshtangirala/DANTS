"""
Genetic feature evolution.

Operates on the already-engineered feature columns (not raw OHLCV). A
"gene" is (feature_a, feature_b, op) where op is applied to feature_a
(and feature_b for binary ops) to derive a new column. Fitness is the
absolute Spearman information coefficient (IC) between the derived
feature and the forward-return label on the pooled training slice.

A small hand-rolled genetic algorithm (no new dependency) evolves a
population of genes over a few generations via tournament selection,
single-point crossover, and random-op mutation, then returns the top-K
winners so they can be appended as extra columns to X (train, val, and
inference alike) and persisted for reproducibility at inference time.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BINARY_OPS = ("ratio", "diff", "product")
UNARY_OPS = ("rolling_mean_5", "rolling_std_5")
ALL_OPS = BINARY_OPS + UNARY_OPS


@dataclass(frozen=True)
class Gene:
    feature_a: str
    feature_b: Optional[str]
    op: str

    def name(self) -> str:
        if self.op in UNARY_OPS:
            return f"evolved_{self.op}_{self.feature_a}"
        return f"evolved_{self.feature_a}_{self.op}_{self.feature_b}"

    def to_dict(self) -> Dict[str, Any]:
        return {"feature_a": self.feature_a, "feature_b": self.feature_b, "op": self.op}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Gene":
        return Gene(feature_a=d["feature_a"], feature_b=d.get("feature_b"), op=d["op"])

    def evaluate(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Compute the derived column, or None if inputs are missing/invalid."""
        if self.feature_a not in df.columns:
            return None

        a = df[self.feature_a].astype(float)

        if self.op in UNARY_OPS:
            if self.op == "rolling_mean_5":
                return a.rolling(window=5, min_periods=1).mean()
            if self.op == "rolling_std_5":
                return a.rolling(window=5, min_periods=1).std().fillna(0.0)
            return None

        if self.feature_b is None or self.feature_b not in df.columns:
            return None
        b = df[self.feature_b].astype(float)

        if self.op == "ratio":
            return a / b.replace(0, np.nan)
        if self.op == "diff":
            return a - b
        if self.op == "product":
            return a * b
        return None


class FeatureEvolver:
    """Small genetic-programming search over derived-feature genes, scored by Spearman IC."""

    def __init__(
        self,
        population_size: int = 30,
        n_generations: int = 15,
        top_k: int = 5,
        tournament_size: int = 4,
        mutation_rate: float = 0.2,
        random_state: int = 42,
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.top_k = top_k
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.rng = np.random.default_rng(random_state)

    def _random_gene(self, feature_cols: List[str]) -> Gene:
        op = self.rng.choice(ALL_OPS)
        feature_a = self.rng.choice(feature_cols)
        if op in UNARY_OPS:
            return Gene(feature_a=feature_a, feature_b=None, op=op)
        remaining = [c for c in feature_cols if c != feature_a]
        feature_b = self.rng.choice(remaining) if remaining else feature_a
        return Gene(feature_a=feature_a, feature_b=feature_b, op=op)

    def _mutate(self, gene: Gene, feature_cols: List[str]) -> Gene:
        if self.rng.random() < self.mutation_rate:
            return self._random_gene(feature_cols)
        return gene

    def _crossover(self, gene_a: Gene, gene_b: Gene) -> Gene:
        """Single-point crossover on binary genes: swap one of (feature_a, feature_b, op)."""
        if gene_a.op in UNARY_OPS or gene_b.op in UNARY_OPS:
            return gene_a if self.rng.random() < 0.5 else gene_b
        pick = self.rng.integers(0, 3)
        if pick == 0:
            return Gene(feature_a=gene_b.feature_a, feature_b=gene_a.feature_b, op=gene_a.op)
        if pick == 1:
            return Gene(feature_a=gene_a.feature_a, feature_b=gene_b.feature_b, op=gene_a.op)
        return Gene(feature_a=gene_a.feature_a, feature_b=gene_a.feature_b, op=gene_b.op)

    @staticmethod
    def _fitness(derived: pd.Series, label: np.ndarray) -> float:
        values = derived.to_numpy()
        finite = np.isfinite(values)
        if finite.sum() < 10:
            return 0.0
        ic, _ = spearmanr(values[finite], label[finite])
        if ic is None or not np.isfinite(ic):
            return 0.0
        return abs(float(ic))

    def evolve(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "future_return",
    ) -> List[Tuple[Gene, float]]:
        """
        Run the GA and return the top_k (gene, fitness) pairs, sorted best-first.
        `label_col` should be a continuous forward-return column (not the
        binarized label) so Spearman IC has more than 2 distinct values to work with.
        """
        if label_col not in df.columns or not feature_cols:
            return []

        label = df[label_col].to_numpy(dtype=float)
        valid_features = [c for c in feature_cols if c in df.columns]
        if len(valid_features) < 1:
            return []

        population = [self._random_gene(valid_features) for _ in range(self.population_size)]

        best_overall: Dict[Gene, float] = {}

        for _generation in range(self.n_generations):
            scored = []
            for gene in population:
                derived = gene.evaluate(df)
                if derived is None:
                    continue
                fitness = self._fitness(derived, label)
                scored.append((gene, fitness))
                if gene not in best_overall or fitness > best_overall[gene]:
                    best_overall[gene] = fitness

            if not scored:
                break

            scored.sort(key=lambda gf: gf[1], reverse=True)

            next_population = [scored[0][0]]  # elitism: keep the best gene
            while len(next_population) < self.population_size:
                tournament_a = self.rng.choice(len(scored), size=min(self.tournament_size, len(scored)), replace=False)
                tournament_b = self.rng.choice(len(scored), size=min(self.tournament_size, len(scored)), replace=False)
                parent_a = max((scored[i] for i in tournament_a), key=lambda gf: gf[1])[0]
                parent_b = max((scored[i] for i in tournament_b), key=lambda gf: gf[1])[0]
                child = self._crossover(parent_a, parent_b)
                child = self._mutate(child, valid_features)
                next_population.append(child)

            population = next_population

        ranked = sorted(best_overall.items(), key=lambda gf: gf[1], reverse=True)
        return ranked[: self.top_k]

    @staticmethod
    def apply_genes(df: pd.DataFrame, genes: List[Gene]) -> pd.DataFrame:
        """Append each gene's derived column to df, filling failures with 0."""
        result = df.copy()
        for gene in genes:
            derived = gene.evaluate(df)
            if derived is None:
                derived = pd.Series(0.0, index=df.index)
            result[gene.name()] = derived.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return result
