"""
RAPO Post-optimization / Refinement Hybrid - POA -> ARO
========================================================

Variant name
------------
RAPO_POR_POA_ARO

Mechanism
---------
1. Run the main optimizer with POA for T iterations.
2. Select the top-k best final individuals from the POA population.
3. For each elite individual, run a short ARO refinement process for L micro-iterations.
4. Replace the original elite only if the refined solution is better.

Problem type
------------
Continuous minimization benchmark problems.

Notes for integration
---------------------
This file is standalone. To integrate with an existing benchmark framework, replace
`_evaluate(objective_func, x)` with the framework's `evaluator.evaluate(x)` and map
`OptimizationResult` to the project's result object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import math
import time

import numpy as np


Array = np.ndarray
ObjectiveFunction = Callable[[Array], float]


@dataclass
class OptimizationResult:
    """Minimal standalone result object for benchmark experiments."""

    algorithm_name: str
    best_solution: Array
    best_fitness: float
    convergence_curve: List[float]
    nfe: int
    runtime_seconds: float
    metadata: Dict[str, object] = field(default_factory=dict)


class RAPOPostOptimizationPOAARO:
    """
    Post-optimization hybrid: POA main search, then ARO refinement on top-k elites.

    Parameters
    ----------
    population_size:
        Number of candidate solutions in the global population.
    max_iterations:
        Number of main POA iterations.
    lower_bound, upper_bound:
        Scalar or vector bounds.
    top_k:
        Number of final elite solutions refined by ARO.
    refinement_iterations:
        Number of ARO micro-iterations per elite.
    refinement_population_size:
        Local ARO population size for each elite refinement. If None, it is inferred.
    refinement_radius:
        Initial local search radius as a fraction of the search range.
    max_evaluations:
        Optional NFE budget. If reached, optimization stops safely.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        population_size: int = 50,
        max_iterations: int = 100,
        lower_bound: float | Array = -100.0,
        upper_bound: float | Array = 100.0,
        top_k: int = 3,
        refinement_iterations: int = 5,
        refinement_population_size: Optional[int] = None,
        refinement_radius: float = 0.10,
        max_evaluations: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if refinement_iterations < 1:
            raise ValueError("refinement_iterations must be at least 1.")
        if refinement_radius <= 0:
            raise ValueError("refinement_radius must be positive.")

        self.population_size = int(population_size)
        self.max_iterations = int(max_iterations)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.top_k = int(top_k)
        self.refinement_iterations = int(refinement_iterations)
        self.refinement_population_size = refinement_population_size
        self.refinement_radius = float(refinement_radius)
        self.max_evaluations = max_evaluations
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.nfe = 0

    def optimize(self, objective_func: ObjectiveFunction, dimension: int) -> OptimizationResult:
        """Run RAPO-POR POA->ARO on a minimization problem."""
        start_time = time.perf_counter()
        self.nfe = 0

        lb, ub = self._prepare_bounds(dimension)
        search_range = ub - lb

        population = self.rng.uniform(lb, ub, size=(self.population_size, dimension))
        fitness = self._evaluate_population(objective_func, population)
        best_idx = int(np.argmin(fitness))
        best_solution = population[best_idx].copy()
        best_fitness = float(fitness[best_idx])

        convergence_curve: List[float] = [best_fitness]
        iteration_trace: List[Dict[str, object]] = []
        refinement_events: List[Dict[str, object]] = []

        # Main phase: POA global search.
        for t in range(1, self.max_iterations + 1):
            if self._budget_exhausted():
                break

            for i in range(self.population_size):
                if self._budget_exhausted():
                    break

                # POA Phase 1: Predator Attack.
                better_indices = np.where(fitness < fitness[i])[0]
                if better_indices.size > 0:
                    sp_idx = int(self.rng.choice(better_indices))
                    selected_pufferfish = population[sp_idx]
                    y = self._poa_attack(population[i], selected_pufferfish, lb, ub)
                    fy = self._evaluate(objective_func, y)
                    if fy <= fitness[i]:
                        population[i] = y
                        fitness[i] = fy

                if self._budget_exhausted():
                    break

                # POA Phase 2: Defense Mechanism.
                z = self._poa_defense(population[i], t, lb, ub)
                fz = self._evaluate(objective_func, z)
                if fz <= fitness[i]:
                    population[i] = z
                    fitness[i] = fz

                if fitness[i] < best_fitness:
                    best_fitness = float(fitness[i])
                    best_solution = population[i].copy()

            convergence_curve.append(best_fitness)
            iteration_trace.append(
                {
                    "stage": "POA_main",
                    "iteration": t,
                    "best_fitness": best_fitness,
                    "nfe": self.nfe,
                }
            )

        # Post-optimization phase: ARO refines top-k final POA elites.
        if not self._budget_exhausted():
            elite_count = min(self.top_k, self.population_size)
            elite_indices = np.argsort(fitness)[:elite_count]

            for rank, idx in enumerate(elite_indices, start=1):
                if self._budget_exhausted():
                    break

                elite_before = population[idx].copy()
                fitness_before = float(fitness[idx])

                refined_solution, refined_fitness, local_nfe = self._aro_refine_elite(
                    objective_func=objective_func,
                    elite=elite_before,
                    elite_fitness=fitness_before,
                    lb=lb,
                    ub=ub,
                    search_range=search_range,
                    elite_rank=rank,
                )

                improved = refined_fitness <= fitness_before
                if improved:
                    population[idx] = refined_solution
                    fitness[idx] = refined_fitness

                if fitness[idx] < best_fitness:
                    best_fitness = float(fitness[idx])
                    best_solution = population[idx].copy()

                convergence_curve.append(best_fitness)
                refinement_events.append(
                    {
                        "elite_rank": rank,
                        "population_index": int(idx),
                        "fitness_before": fitness_before,
                        "fitness_after": float(fitness[idx]),
                        "improved": bool(improved),
                        "local_nfe": int(local_nfe),
                        "global_nfe": int(self.nfe),
                    }
                )

        runtime = time.perf_counter() - start_time
        return OptimizationResult(
            algorithm_name="RAPO_POR_POA_ARO",
            best_solution=best_solution,
            best_fitness=best_fitness,
            convergence_curve=convergence_curve,
            nfe=self.nfe,
            runtime_seconds=runtime,
            metadata={
                "hybrid_type": "Post-optimization / Refinement Hybrid",
                "main_optimizer": "POA",
                "refinement_optimizer": "ARO",
                "top_k": self.top_k,
                "refinement_iterations": self.refinement_iterations,
                "refinement_population_size": self._local_population_size(),
                "refinement_radius": self.refinement_radius,
                "max_iterations": self.max_iterations,
                "population_size": self.population_size,
                "max_evaluations": self.max_evaluations,
                "seed": self.seed,
                "nfe": self.nfe,
                "refinement_events": refinement_events,
                "iteration_trace": iteration_trace,
            },
        )

    # ------------------------- POA operators -------------------------

    def _poa_attack(self, xi: Array, selected_pufferfish: Array, lb: Array, ub: Array) -> Array:
        dim = xi.size
        r = self.rng.random(dim)
        integer_factor = self.rng.integers(1, 3, size=dim)
        candidate = xi + r * (selected_pufferfish - integer_factor * xi)
        return self._repair(candidate, lb, ub)

    def _poa_defense(self, xi: Array, t: int, lb: Array, ub: Array) -> Array:
        dim = xi.size
        r = self.rng.random(dim)
        candidate = xi + (1.0 - 2.0 * r) * (ub - lb) / max(t, 1)
        return self._repair(candidate, lb, ub)

    # ------------------------- ARO refinement -------------------------

    def _aro_refine_elite(
        self,
        objective_func: ObjectiveFunction,
        elite: Array,
        elite_fitness: float,
        lb: Array,
        ub: Array,
        search_range: Array,
        elite_rank: int,
    ) -> Tuple[Array, float, int]:
        """Run a short local ARO search around one elite solution."""
        local_nfe_start = self.nfe
        local_size = self._local_population_size()
        dim = elite.size

        radius = self.refinement_radius * search_range
        local_population = elite + self.rng.normal(0.0, 1.0, size=(local_size, dim)) * radius
        local_population = np.clip(local_population, lb, ub)
        local_population[0] = elite.copy()

        local_fitness = self._evaluate_population(objective_func, local_population)

        for micro_t in range(1, self.refinement_iterations + 1):
            if self._budget_exhausted():
                break

            energy = self._aro_energy(micro_t, self.refinement_iterations)
            radius_scale = max(1e-12, 1.0 - (micro_t - 1) / self.refinement_iterations)

            for i in range(local_size):
                if self._budget_exhausted():
                    break

                if energy > 1.0:
                    y = self._aro_detour_foraging_around_elite(
                        local_population, i, elite, micro_t, self.refinement_iterations, radius_scale, lb, ub
                    )
                else:
                    y = self._aro_random_hiding_around_elite(
                        local_population[i], elite, micro_t, self.refinement_iterations, radius_scale, lb, ub
                    )

                fy = self._evaluate(objective_func, y)
                if fy <= local_fitness[i]:
                    local_population[i] = y
                    local_fitness[i] = fy

        local_best_idx = int(np.argmin(local_fitness))
        refined_solution = local_population[local_best_idx].copy()
        refined_fitness = float(local_fitness[local_best_idx])

        if elite_fitness <= refined_fitness:
            return elite.copy(), float(elite_fitness), self.nfe - local_nfe_start
        return refined_solution, refined_fitness, self.nfe - local_nfe_start

    def _aro_energy(self, t: int, max_t: int) -> float:
        r = float(self.rng.uniform(1e-12, 1.0))
        return 4.0 * (1.0 - t / max_t) * math.log(1.0 / r)

    def _aro_detour_foraging_around_elite(
        self,
        population: Array,
        i: int,
        elite: Array,
        t: int,
        max_t: int,
        radius_scale: float,
        lb: Array,
        ub: Array,
    ) -> Array:
        n, dim = population.shape
        j = int(self.rng.integers(0, n - 1))
        if j >= i:
            j += 1

        r1 = float(self.rng.random())
        r2 = float(self.rng.random())
        r3 = float(self.rng.random())
        normal_noise = self.rng.normal(0.0, 1.0, size=dim)

        running_length = (math.e - math.exp(((t - 1) / max_t) ** 2)) * math.sin(2.0 * math.pi * r2)
        mapping = np.zeros(dim)
        selected_dims = max(1, int(math.ceil(r3 * dim)))
        mapping[self.rng.permutation(dim)[:selected_dims]] = 1.0
        running_operator = running_length * mapping

        perturbation_switch = round(0.5 * (0.05 + r1))
        candidate = population[j] + running_operator * (population[i] - population[j]) + perturbation_switch * normal_noise
        candidate = elite + radius_scale * (candidate - elite)
        return self._repair(candidate, lb, ub)

    def _aro_random_hiding_around_elite(
        self,
        individual: Array,
        elite: Array,
        t: int,
        max_t: int,
        radius_scale: float,
        lb: Array,
        ub: Array,
    ) -> Array:
        dim = individual.size
        random_dim = int(self.rng.integers(0, dim))
        gr = np.zeros(dim)
        gr[random_dim] = 1.0

        r4 = float(self.rng.random())
        hiding_parameter = ((max_t - t + 1.0) / max_t) * r4
        burrow = individual + hiding_parameter * gr * individual

        running_length = (math.e - math.exp(((t - 1) / max_t) ** 2)) * math.sin(2.0 * math.pi * self.rng.random())
        candidate = individual + running_length * (r4 * burrow - individual)
        candidate = elite + radius_scale * (candidate - elite)
        return self._repair(candidate, lb, ub)

    # ------------------------- utilities -------------------------

    def _local_population_size(self) -> int:
        if self.refinement_population_size is not None:
            return max(2, int(self.refinement_population_size))
        return max(5, min(self.population_size, 2 * self.top_k + 3))

    def _prepare_bounds(self, dimension: int) -> Tuple[Array, Array]:
        lb = np.asarray(self.lower_bound, dtype=float)
        ub = np.asarray(self.upper_bound, dtype=float)

        if lb.ndim == 0:
            lb = np.full(dimension, float(lb))
        if ub.ndim == 0:
            ub = np.full(dimension, float(ub))
        if lb.shape != (dimension,) or ub.shape != (dimension,):
            raise ValueError("lower_bound and upper_bound must be scalar or arrays of shape (dimension,).")
        if np.any(lb >= ub):
            raise ValueError("Each lower_bound must be smaller than upper_bound.")
        return lb, ub

    @staticmethod
    def _repair(x: Array, lb: Array, ub: Array) -> Array:
        return np.clip(x, lb, ub)

    def _evaluate(self, objective_func: ObjectiveFunction, x: Array) -> float:
        if self._budget_exhausted():
            return float("inf")
        value = float(objective_func(np.asarray(x, dtype=float)))
        self.nfe += 1
        return value

    def _evaluate_population(self, objective_func: ObjectiveFunction, population: Array) -> Array:
        values = np.empty(population.shape[0], dtype=float)
        for i, individual in enumerate(population):
            values[i] = self._evaluate(objective_func, individual)
            if self._budget_exhausted():
                values[i + 1 :] = float("inf")
                break
        return values

    def _budget_exhausted(self) -> bool:
        return self.max_evaluations is not None and self.nfe >= self.max_evaluations


if __name__ == "__main__":
    def sphere(x: Array) -> float:
        return float(np.sum(x ** 2))

    optimizer = RAPOPostOptimizationPOAARO(
        population_size=30,
        max_iterations=50,
        lower_bound=-100,
        upper_bound=100,
        top_k=3,
        refinement_iterations=5,
        seed=42,
    )
    result = optimizer.optimize(sphere, dimension=10)
    print("Algorithm:", result.algorithm_name)
    print("Best fitness:", result.best_fitness)
    print("NFE:", result.nfe)
    print("Runtime:", round(result.runtime_seconds, 6), "seconds")
    print("Refinement events:", result.metadata["refinement_events"])
