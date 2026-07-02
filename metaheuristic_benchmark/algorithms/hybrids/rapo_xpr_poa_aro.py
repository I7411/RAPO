"""
RAPO phase-replacement variants for continuous minimization benchmarks.

These files are written as standalone Python 3.11 modules so they can be tested
without the full benchmark framework. When integrating into your framework, the
main adaptation is usually replacing `_evaluate(...)` with `evaluator.evaluate(...)`
and returning your project's OptimizationResult class.

Core assumptions:
- Minimization problem.
- Population is continuous: X shape = (population_size, dimension).
- Boundary repair uses clipping to [lower_bound, upper_bound].
- All objective-function calls are counted through `nfe`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import math
import time
import numpy as np

ArrayLikeBound = Union[float, int, List[float], Tuple[float, ...], np.ndarray]
ObjectiveFunction = Callable[[np.ndarray], float]


@dataclass
class OptimizationResult:
    best_solution: np.ndarray
    best_fitness: float
    convergence_curve: List[float]
    nfe: int
    runtime_seconds: float
    metadata: Dict[str, object] = field(default_factory=dict)


class _RAPOBase:
    """Shared implementation of ARO and Pufferfish-POA operators."""

    def __init__(
        self,
        population_size: int = 50,
        max_iterations: int = 100,
        lower_bound: ArrayLikeBound = -100.0,
        upper_bound: ArrayLikeBound = 100.0,
        energy_threshold: float = 1.0,
        seed: Optional[int] = None,
        max_evaluations: Optional[int] = None,
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        if max_evaluations is not None and max_evaluations < population_size:
            raise ValueError("max_evaluations should be >= population_size for initialization.")

        self.population_size = int(population_size)
        self.max_iterations = int(max_iterations)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.energy_threshold = float(energy_threshold)
        self.seed = seed
        self.max_evaluations = max_evaluations

        self.rng = np.random.default_rng(seed)
        self.nfe = 0
        self._lb: np.ndarray
        self._ub: np.ndarray
        self._best_idx = 0

    def _prepare_bounds(self, dimension: int) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.asarray(self.lower_bound, dtype=float)
        ub = np.asarray(self.upper_bound, dtype=float)

        if lb.ndim == 0:
            lb = np.full(dimension, float(lb))
        if ub.ndim == 0:
            ub = np.full(dimension, float(ub))

        if lb.shape != (dimension,) or ub.shape != (dimension,):
            raise ValueError("Bounds must be scalar or arrays with shape (dimension,).")
        if np.any(lb >= ub):
            raise ValueError("Each lower_bound must be smaller than upper_bound.")
        return lb, ub

    def _evaluate(self, objective_func: ObjectiveFunction, x: np.ndarray) -> float:
        if self.max_evaluations is not None and self.nfe >= self.max_evaluations:
            raise StopIteration("Evaluation budget exhausted.")
        value = float(objective_func(np.asarray(x, dtype=float)))
        self.nfe += 1
        return value

    def _repair(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=float), self._lb, self._ub)

    def _initialize_population(
        self,
        objective_func: ObjectiveFunction,
        dimension: int,
        initial_population: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if initial_population is None:
            population = self.rng.uniform(self._lb, self._ub, size=(self.population_size, dimension))
        else:
            population = np.asarray(initial_population, dtype=float)
            if population.shape != (self.population_size, dimension):
                raise ValueError(
                    "initial_population must have shape "
                    f"({self.population_size}, {dimension})."
                )
            population = self._repair(population)

        fitness = np.empty(self.population_size, dtype=float)
        for i in range(self.population_size):
            fitness[i] = self._evaluate(objective_func, population[i])
        self._best_idx = int(np.argmin(fitness))
        return population, fitness

    def _update_best(self, fitness: np.ndarray) -> int:
        self._best_idx = int(np.argmin(fitness))
        return self._best_idx

    def _aro_energy(self, t: int) -> float:
        # A(t) = 4 * (1 - t/T) * ln(1/r), r in (0, 1)
        r = float(np.clip(self.rng.random(), 1e-12, 1.0 - 1e-12))
        return 4.0 * (1.0 - t / self.max_iterations) * math.log(1.0 / r)

    def _aro_running_operator(self, dimension: int, t: int) -> np.ndarray:
        # Running length L and mapping vector c from ARO detour/hiding equations.
        r2 = self.rng.random()
        r3 = self.rng.random()
        length = (math.e - math.exp(((t - 1.0) / self.max_iterations) ** 2.0)) * math.sin(2.0 * math.pi * r2)

        c = np.zeros(dimension, dtype=float)
        count = max(1, int(math.ceil(r3 * dimension)))
        selected = self.rng.permutation(dimension)[:count]
        c[selected] = 1.0
        return length * c

    def _aro_detour_foraging(self, i: int, population: np.ndarray, t: int) -> np.ndarray:
        """ARO exploration operator: Detour Foraging."""
        n, dimension = population.shape
        j = int(self.rng.integers(0, n - 1))
        if j >= i:
            j += 1

        running = self._aro_running_operator(dimension, t)
        perturb_gate = round(0.5 * (0.05 + self.rng.random()))
        gaussian_noise = self.rng.normal(0.0, 1.0, size=dimension)

        candidate = (
            population[j]
            + running * (population[i] - population[j])
            + perturb_gate * gaussian_noise
        )
        return candidate

    def _aro_random_hiding(self, i: int, population: np.ndarray, t: int) -> np.ndarray:
        """ARO exploitation operator: Random Hiding."""
        _, dimension = population.shape
        hiding_factor = ((self.max_iterations - t + 1.0) / self.max_iterations) * self.rng.random()

        one_hot = np.zeros(dimension, dtype=float)
        one_hot[int(self.rng.integers(0, dimension))] = 1.0
        random_burrow = population[i] + hiding_factor * one_hot * population[i]

        running = self._aro_running_operator(dimension, t)
        candidate = population[i] + running * (self.rng.random() * random_burrow - population[i])
        return candidate

    def _poa_predator_attack(self, i: int, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Pufferfish-POA Phase 1: Predator Attack towards Pufferfish.

        Candidate pufferfish CP_i are members better than X_i. If CP_i is empty,
        the current global best is used as a safe fallback so the operator still
        returns a candidate while greedy selection protects against deterioration.
        """
        better_indices = np.flatnonzero((fitness < fitness[i]) & (np.arange(len(fitness)) != i))
        if better_indices.size > 0:
            selected_idx = int(self.rng.choice(better_indices))
        else:
            selected_idx = self._best_idx

        selected_pufferfish = population[selected_idx]
        r = self.rng.random(population.shape[1])
        intensity = self.rng.integers(1, 3, size=population.shape[1])  # I in {1, 2}
        candidate = population[i] + r * (selected_pufferfish - intensity * population[i])
        return candidate

    def _poa_defense_mechanism(self, i: int, population: np.ndarray, t: int) -> np.ndarray:
        """Pufferfish-POA Phase 2: Defense Mechanism exploitation operator."""
        r = self.rng.random(population.shape[1])
        step = (1.0 - 2.0 * r) * ((self._ub - self._lb) / max(t, 1))
        candidate = population[i] + step
        return candidate

    def _greedy_update(
        self,
        objective_func: ObjectiveFunction,
        population: np.ndarray,
        fitness: np.ndarray,
        i: int,
        candidate: np.ndarray,
    ) -> bool:
        candidate = self._repair(candidate)
        candidate_fitness = self._evaluate(objective_func, candidate)
        if candidate_fitness <= fitness[i]:
            population[i] = candidate
            fitness[i] = candidate_fitness
            return True
        return False


class RAPOExploitationReplacementPOAARO(_RAPOBase):
    """RAPO-XPR POA->ARO.

    Low-level Relay / Exploitation Phase Replacement.
    Keep the Pufferfish-POA two-phase framework.
    Keep POA Phase 1 Predator Attack.
    Replace POA Phase 2 Defense Mechanism with ARO Random Hiding.

    Each individual may trigger up to two objective evaluations per iteration.
    Use max_evaluations for fair comparison against one-evaluation-per-individual variants.
    """

    def optimize(
        self,
        objective_func: ObjectiveFunction,
        dimension: int,
        initial_population: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        self.rng = np.random.default_rng(self.seed)
        self.nfe = 0
        self._lb, self._ub = self._prepare_bounds(dimension)
        start_time = time.perf_counter()

        population, fitness = self._initialize_population(objective_func, dimension, initial_population)
        best_idx = self._update_best(fitness)
        convergence_curve = [float(fitness[best_idx])]
        operator_counter = {"poa_predator_attack": 0, "aro_random_hiding": 0}

        stop_reason = "max_iterations"
        try:
            for t in range(1, self.max_iterations + 1):
                for i in range(self.population_size):
                    candidate_1 = self._poa_predator_attack(i, population, fitness)
                    operator_counter["poa_predator_attack"] += 1
                    self._greedy_update(objective_func, population, fitness, i, candidate_1)
                    self._update_best(fitness)

                    candidate_2 = self._aro_random_hiding(i, population, t)
                    operator_counter["aro_random_hiding"] += 1
                    self._greedy_update(objective_func, population, fitness, i, candidate_2)
                    self._update_best(fitness)

                best_idx = self._update_best(fitness)
                convergence_curve.append(float(fitness[best_idx]))
        except StopIteration:
            stop_reason = "max_evaluations"

        best_idx = self._update_best(fitness)
        runtime_seconds = time.perf_counter() - start_time
        return OptimizationResult(
            best_solution=population[best_idx].copy(),
            best_fitness=float(fitness[best_idx]),
            convergence_curve=convergence_curve,
            nfe=self.nfe,
            runtime_seconds=runtime_seconds,
            metadata={
                "algorithm": "RAPO_XPR_POA_ARO",
                "hybrid_type": "Low-level Relay / Exploitation Phase Replacement",
                "framework": "POA",
                "kept_operator": "POA Phase 1 - Predator Attack",
                "replaced_phase": "POA Phase 2 - Defense Mechanism",
                "replacement_operator": "ARO Random Hiding",
                "operator_counter": operator_counter,
                "stop_reason": stop_reason,
            },
        )


if __name__ == "__main__":
    def sphere(x: np.ndarray) -> float:
        return float(np.sum(x ** 2))

    optimizer = RAPOExploitationReplacementPOAARO(
        population_size=30,
        max_iterations=80,
        lower_bound=-100.0,
        upper_bound=100.0,
        seed=42,
    )
    result = optimizer.optimize(sphere, dimension=30)

    print("algorithm:", result.metadata["algorithm"])
    print("best_fitness:", result.best_fitness)
    print("nfe:", result.nfe)
    print("runtime_seconds:", result.runtime_seconds)
    print("last_convergence:", result.convergence_curve[-5:])
