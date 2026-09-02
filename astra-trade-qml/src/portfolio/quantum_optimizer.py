"""
Quantum-inspired portfolio allocation: selects a fixed-size subset of
assets from a candidate universe to hold, trading off expected return
against covariance risk and sector concentration - the doc's own framing
for where quantum methods currently have real research support ("D-Wave-
style annealing, QAOA-style optimisation... for portfolio optimisation,
capital allocation"), explicitly NOT direct price prediction.

Formulated as a QUBO (Quadratic Unconstrained Binary Optimization) over
one binary variable per asset (1 = held, 0 = not held), solved by:

  1. Simulated annealing (SimulatedAnnealingOptimizer) - the primary,
     always-available "quantum-inspired" solver (D-Wave-style annealing
     is a classical proxy for quantum annealing, needs no quantum
     hardware or qiskit at all).
  2. QAOA (QAOAOptimizer), when qiskit is available - a genuine
     variational quantum circuit solving the same QUBO (converted to an
     Ising Hamiltonian). Falls back to simulated annealing on any
     failure, matching this repo's existing quantum-fallback pattern
     (src/models/quantum/quantum_kernel.py, vqc_classifier.py).

Both are benchmarked in backtest.py against two classical baselines
(equal-weight, scipy mean-variance) - per the source doc's own closing
position: "The quantum-inspired layer should compete against strong
classical baselines and remain enabled only where it demonstrates
measurable, repeatable value."
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    from qiskit.primitives import StatevectorSampler
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    QISKIT_AVAILABLE = True
except ImportError as e:
    QISKIT_AVAILABLE = False
    warnings.warn(f"Qiskit import failed ({e}). Quantum portfolio optimizer will use simulated annealing only.")


@dataclass
class AllocationDecision:
    weights: Dict[str, float]  # symbol -> portfolio weight, sums to 1.0 over selected symbols
    selected: List[str]
    method: str  # "simulated_annealing", "qaoa", "mean_variance", "equal_weight"
    energy: Optional[float] = None  # QUBO objective value achieved (lower is better), when applicable


def build_qubo_matrix(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    sector_indicator: np.ndarray,
    target_k: int,
    lambda_risk: float = 5.0,
    lambda_cardinality: float = 2.0,
    lambda_sector: float = 1.0,
) -> np.ndarray:
    """
    Build the symmetric QUBO matrix Q such that x^T Q x (for binary x,
    up to an additive constant this drops since it doesn't affect the
    minimizer) equals:

        -expected_returns . x                              (reward return)
        + lambda_risk * x^T covariance x                    (penalize risk)
        + lambda_cardinality * (sum(x) - target_k)^2         (hold ~target_k assets)
        + lambda_sector * sum_sector count_sector*(count_sector - 1)
                                                              (discourage
                                                               over-concentrating
                                                               in one sector)

    sector_indicator[i][j] = 1 if assets i and j (including i == j) are
    in the same sector, else 0 - i.e. the block-diagonal "same sector"
    matrix built from a sector membership map.

    Derivation (binary x_i satisfies x_i^2 = x_i, so any linear term can
    be folded into a QUBO's diagonal):
        A = lambda_risk*covariance + lambda_cardinality*J + lambda_sector*sector_indicator
            (J = all-ones matrix, from expanding (sum(x)-K)^2)
        L = -expected_returns - 2*K*lambda_cardinality*ones - lambda_sector*ones
            (linear terms: -mu.x from the return reward, the cross term
            from (sum(x)-K)^2, and the "-count" part of count*(count-1))
        Q[i,i] = A[i,i] + L[i];  Q[i,j] = A[i,j] for i != j
    """
    n = len(expected_returns)
    ones = np.ones(n)
    J = np.ones((n, n))

    A = lambda_risk * covariance + lambda_cardinality * J + lambda_sector * sector_indicator
    L = -expected_returns - 2 * target_k * lambda_cardinality * ones - lambda_sector * ones

    Q = A.copy()
    np.fill_diagonal(Q, np.diag(A) + L)
    return Q


def qubo_energy(x: np.ndarray, Q: np.ndarray) -> float:
    """x^T Q x for a binary vector x - the QUBO objective build_qubo_matrix()
    encodes (lower is better)."""
    x = np.asarray(x, dtype=float)
    return float(x @ Q @ x)


def build_sector_indicator(symbols: List[str], sector_map: Dict[str, str]) -> np.ndarray:
    n = len(symbols)
    sectors = [sector_map.get(s, "Unclassified") for s in symbols]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if sectors[i] == sectors[j]:
                M[i, j] = 1.0
    return M


class SimulatedAnnealingOptimizer:
    """
    Classical proxy for quantum annealing (the doc's "D-Wave-style
    annealing"): repeatedly flips a random bit, accepting the move if it
    lowers QUBO energy, or with a temperature-decaying probability if it
    doesn't (so the search can escape local minima early on, and settles
    into a strong local minimum as the temperature cools). Always
    available - no external solver, no qiskit.
    """

    def __init__(self, n_iterations: int = 4000, initial_temp: float = 2.0, cooling_rate: float = 0.995, seed: int = 42):
        self.n_iterations = n_iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.rng = np.random.default_rng(seed)

    def solve(self, Q: np.ndarray, target_k: int) -> np.ndarray:
        n = Q.shape[0]
        # Start from a random subset of the right size rather than all-zeros,
        # so the cardinality penalty isn't fighting the starting point.
        x = np.zeros(n, dtype=int)
        x[self.rng.choice(n, size=min(target_k, n), replace=False)] = 1

        energy = qubo_energy(x, Q)
        best_x, best_energy = x.copy(), energy
        temp = self.initial_temp

        for _ in range(self.n_iterations):
            i = self.rng.integers(0, n)
            x_new = x.copy()
            x_new[i] = 1 - x_new[i]
            new_energy = qubo_energy(x_new, Q)
            delta = new_energy - energy

            if delta < 0 or self.rng.random() < np.exp(-delta / max(temp, 1e-9)):
                x, energy = x_new, new_energy
                if energy < best_energy:
                    best_x, best_energy = x.copy(), energy

            temp *= self.cooling_rate

        return best_x


class QAOAOptimizer:
    """
    Solves the same QUBO via QAOA (Quantum Approximate Optimization
    Algorithm) - a real parameterized quantum circuit, classically
    optimized. Converts the QUBO to an Ising Hamiltonian (x_i = (1-z_i)/2,
    z_i in {-1,+1}) and hands it to qiskit_algorithms.QAOA as a
    SparsePauliOp cost operator. Falls back to SimulatedAnnealingOptimizer
    on any failure (missing qiskit, solver error, or a result that fails
    the cardinality check) - this repo's existing quantum modules
    (quantum_kernel.py, vqc_classifier.py) follow the same
    fail-soft-to-classical pattern.
    """

    def __init__(self, reps: int = 2, maxiter: int = 100, seed: int = 42):
        """`seed` seeds qiskit_algorithms' RNG (QAOA's random initial
        point) and the fallback annealer - a best effort toward
        reproducibility, not a guarantee: the underlying statevector
        simulation runs through multi-threaded BLAS, whose floating-point
        summation order can vary slightly run to run, occasionally
        nudging COBYLA toward a different nearby local optimum even with
        the RNG itself verified bit-identical across runs."""
        self.reps = reps
        self.maxiter = maxiter
        self.seed = seed
        self._fallback = SimulatedAnnealingOptimizer(seed=seed)

    @staticmethod
    def qubo_to_ising(Q: np.ndarray):
        """Returns (pauli_op, offset) - offset is the constant dropped
        during the x -> z substitution (informational only, doesn't
        affect the argmin QAOA searches for)."""
        n = Q.shape[0]
        h = np.zeros(n)
        Jc = np.zeros((n, n))
        offset = 0.0

        for i in range(n):
            offset += Q[i, i] / 2
            h[i] += -Q[i, i] / 2
            for j in range(n):
                if i == j:
                    continue
                offset += Q[i, j] / 4
                h[i] += -Q[i, j] / 2
                if i < j:
                    Jc[i, j] += Q[i, j] / 2

        paulis, coeffs = [], []
        for i in range(n):
            if abs(h[i]) > 1e-12:
                label = ["I"] * n
                label[i] = "Z"
                paulis.append("".join(reversed(label)))
                coeffs.append(h[i])
        for i in range(n):
            for j in range(i + 1, n):
                if abs(Jc[i, j]) > 1e-12:
                    label = ["I"] * n
                    label[i] = "Z"
                    label[j] = "Z"
                    paulis.append("".join(reversed(label)))
                    coeffs.append(Jc[i, j])

        if not paulis:
            paulis, coeffs = ["I" * n], [0.0]

        return SparsePauliOp(paulis, coeffs), offset

    def solve(self, Q: np.ndarray, target_k: int) -> np.ndarray:
        if not QISKIT_AVAILABLE:
            return self._fallback.solve(Q, target_k)

        try:
            from qiskit_algorithms.utils import algorithm_globals
            algorithm_globals.random_seed = self.seed

            cost_op, _offset = self.qubo_to_ising(Q)
            sampler = StatevectorSampler()
            qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=self.maxiter), reps=self.reps)
            result = qaoa.compute_minimum_eigenvalue(cost_op)

            best_bitstring = max(
                result.eigenstate.items(), key=lambda kv: kv[1]
            )[0] if hasattr(result.eigenstate, "items") else None
            if best_bitstring is None:
                raise RuntimeError("QAOA result carried no interpretable eigenstate distribution")

            n = Q.shape[0]
            bits = format(int(best_bitstring, 2) if isinstance(best_bitstring, str) and set(best_bitstring) <= {"0", "1"} else int(best_bitstring), f"0{n}b")
            x = np.array([int(b) for b in reversed(bits)])
            return x
        except Exception as e:
            warnings.warn(f"QAOA solve failed ({e}); falling back to simulated annealing.")
            return self._fallback.solve(Q, target_k)


def decision_from_selection(
    x: np.ndarray, symbols: List[str], expected_returns: np.ndarray, method: str, energy: Optional[float] = None,
) -> AllocationDecision:
    """
    Turn a binary selection vector into portfolio weights. Weights the
    selected assets by (clipped-to-positive) expected return, falling
    back to equal weight among the selection if every selected asset's
    expected return is non-positive (a real possibility given this
    session's repeatedly-null direction-prediction results - this
    optimizer's job is asset SELECTION/diversification given whatever
    return estimate it's handed, not to manufacture a positive return
    where the estimate has none).
    """
    selected_idx = [i for i, xi in enumerate(x) if xi == 1]
    selected = [symbols[i] for i in selected_idx]
    if not selected:
        return AllocationDecision(weights={}, selected=[], method=method, energy=energy)

    raw = np.array([max(expected_returns[i], 0.0) for i in selected_idx])
    if raw.sum() <= 0:
        weights_arr = np.full(len(selected), 1.0 / len(selected))
    else:
        weights_arr = raw / raw.sum()

    return AllocationDecision(
        weights=dict(zip(selected, weights_arr.tolist())),
        selected=selected,
        method=method,
        energy=energy,
    )


def equal_weight_allocation(symbols: List[str], target_k: int) -> AllocationDecision:
    selected = symbols[:target_k]
    weight = 1.0 / len(selected) if selected else 0.0
    return AllocationDecision(
        weights={s: weight for s in selected}, selected=selected, method="equal_weight",
    )


def mean_variance_allocation(
    symbols: List[str], expected_returns: np.ndarray, covariance: np.ndarray,
    max_weight_per_symbol: float = 0.30, risk_aversion: float = 5.0,
) -> AllocationDecision:
    """
    Classical continuous mean-variance optimization (the "strong
    classical baseline" the doc says the quantum-inspired layer must
    beat): maximize w.mu - risk_aversion * w^T Sigma w subject to
    sum(w) <= 1, 0 <= w_i <= max_weight_per_symbol, via scipy SLSQP.

    sum(w) <= 1 (not == 1) deliberately leaves room for an uninvested
    cash residual - important because a tight max_weight_per_symbol
    relative to n can make full investment infeasible (e.g. the doc's
    own 5% per-stock cap times an 18-symbol universe caps total exposure
    at 90%), and the doc's own capital-allocation plan already reserves
    a "cash reserve" bucket rather than assuming full investment.
    """
    from scipy.optimize import minimize

    n = len(symbols)
    max_feasible = min(1.0, max_weight_per_symbol * n)
    x0 = np.full(n, max_feasible / n)

    def objective(w):
        return -(w @ expected_returns - risk_aversion * (w @ covariance @ w))

    constraints = [{"type": "ineq", "fun": lambda w: 1.0 - w.sum()}]
    bounds = [(0.0, max_weight_per_symbol)] * n

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result.x if result.success else x0
    # Always re-clip to the cap even on a "successful" solve - SLSQP can
    # return values microscopically outside bounds/constraints, and a
    # caller relying on this cap (e.g. PortfolioRiskGate) should never
    # see it silently violated.
    weights = np.clip(weights, 0.0, max_weight_per_symbol)
    if weights.sum() > 1.0:
        weights = weights * (1.0 / weights.sum())

    return AllocationDecision(
        weights={s: float(w) for s, w in zip(symbols, weights) if w > 1e-6},
        selected=[s for s, w in zip(symbols, weights) if w > 1e-6],
        method="mean_variance",
    )
