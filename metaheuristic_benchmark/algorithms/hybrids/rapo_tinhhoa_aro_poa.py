"""
RAPO Elite-guided Hybridization: ARO -> POA
================================================

Variant name:
    RAPO_EGH_ARO_POA

Hybrid mechanism:
    Elite-guided Hybridization.
    ARO is used as the main global optimizer to discover promising elites.
    After every M iterations, the top-k ARO elites are refined locally by POA
    Defense Mechanism for L micro-iterations.

Problem type:
    Continuous minimization benchmark functions.

Design notes:
    - Standalone implementation for easy testing.
    - Uses NumPy only.
    - Tracks NFE (number of function evaluations).
    - Supports max_evaluations for fair comparison.
    - Can be adapted to a framework BaseOptimizer/Evaluator by replacing
      self._evaluate(objective_func, x) with evaluator.evaluate(x).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


Array = np.ndarray
ObjectiveFunc = Callable[[Array], float]


@dataclass
class OptimizationResult:
    """Simple standalone result container."""

    best_solution: Array
    best_fitness: float
    convergence_curve: List[float]
    nfe: int
    runtime_seconds: float
    metadata: Dict = field(default_factory=dict)


class RAPOEliteGuidedAROPOA:
    """
    RAPO-EGH variant: ARO finds elites, POA refines around ARO elites.

    Main loop:
        1. Evolve the whole population using ARO.
        2. Save global best.
        3. If t mod M == 0, select top-k elites from the ARO population.
        4. For each elite, run POA local refinement / Phase 2 for L micro-steps.
        5. Greedy replace the elite if the refined solution is better.

    Parameters
    ----------
    population_size:
        Number of candidate solutions.
    max_iterations:
        Maximum number of outer iterations.
    lower_bound, upper_bound:
        Scalar or vector bounds.
    top_k:
        Number of elites refined at each refinement event.
    refinement_interval:
        M. Apply POA refinement every M iterations.
    local_iterations:
        L. Number of POA micro-iterations for each selected elite.
    seed:
        Random seed for reproducibility.
    max_evaluations:
        Optional NFE budget stop condition.
    """

    def __init__(
        self,
        population_size: int = 50,
        max_iterations: int = 100,
        lower_bound: float | Array = -100.0,
        upper_bound: float | Array = 100.0,
        top_k: int = 3,
        refinement_interval: int = 10,
        local_iterations: int = 5,
        seed: Optional[int] = None,
        max_evaluations: Optional[int] = None,
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size must be >= 2")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if refinement_interval < 1:
            raise ValueError("refinement_interval must be >= 1")
        if local_iterations < 1:
            raise ValueError("local_iterations must be >= 1")

        self.population_size = int(population_size)
        self.max_iterations = int(max_iterations)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.top_k = int(top_k)
        self.refinement_interval = int(refinement_interval)
        self.local_iterations = int(local_iterations)
        self.seed = seed
        self.max_evaluations = max_evaluations

        self.rng = np.random.default_rng(seed)
        self.nfe = 0
        self._budget_exhausted = False

    def optimize(self, objective_func: ObjectiveFunc, dimension: int) -> OptimizationResult:
        """Run RAPO-EGH ARO -> POA on a minimization problem."""
        if dimension < 1:
            raise ValueError("dimension must be >= 1")

        start_time = perf_counter()
        lb, ub = self._prepare_bounds(dimension)
        self.nfe = 0
        self._budget_exhausted = False

        population = self.rng.uniform(lb, ub, size=(self.population_size, dimension))
        fitness = np.array([self._evaluate(objective_func, x) for x in population])

        best_idx = int(np.argmin(fitness))
        best_solution = population[best_idx].copy()
        best_fitness = float(fitness[best_idx])

        convergence_curve: List[float] = [best_fitness]
        refinement_events: List[Dict] = []
        iteration_trace: List[Dict] = []

        for t in range(1, self.max_iterations + 1):
            if self._should_stop():
                break

            # Main ARO evolution.
            improved_by_aro = 0
            for i in range(self.population_size):
                if self._should_stop():
                    break

                energy = self._aro_energy(t)
                if energy > 1.0:
                    candidate = self._aro_detour(population, i, t, lb, ub)
                    operator_name = "ARO_Detour_Foraging"
                else:
                    candidate = self._aro_random_hiding(population[i], t, lb, ub)
                    operator_name = "ARO_Random_Hiding"

                candidate = self._repair(candidate, lb, ub)
                candidate_fitness = self._evaluate(objective_func, candidate)

                if candidate_fitness <= fitness[i]:
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    improved_by_aro += 1

                if fitness[i] < best_fitness:
                    best_fitness = float(fitness[i])
                    best_solution = population[i].copy()

            # Elite-guided POA local refinement around top-k ARO elites.
            refined_count = 0
            if not self._should_stop() and t % self.refinement_interval == 0:
                event_before = best_fitness
                refined_count = self._poa_refine_top_elites(
                    objective_func=objective_func,
                    population=population,
                    fitness=fitness,
                    lb=lb,
                    ub=ub,
                    outer_iteration=t,
                )

                best_idx = int(np.argmin(fitness))
                if fitness[best_idx] < best_fitness:
                    best_fitness = float(fitness[best_idx])
                    best_solution = population[best_idx].copy()

                refinement_events.append(
                    {
                        "iteration": t,
                        "mode": "ARO_elite_to_POA_refinement",
                        "top_k": min(self.top_k, self.population_size),
                        "local_iterations": self.local_iterations,
                        "refined_elites": refined_count,
                        "best_before": event_before,
                        "best_after": best_fitness,
                    }
                )

            convergence_curve.append(best_fitness)
            iteration_trace.append(
                {
                    "iteration": t,
                    "best_fitness": best_fitness,
                    "nfe": self.nfe,
                    "aro_improvements": improved_by_aro,
                    "poa_refined_elites": refined_count,
                }
            )

        runtime = perf_counter() - start_time
        return OptimizationResult(
            best_solution=best_solution,
            best_fitness=best_fitness,
            convergence_curve=convergence_curve,
            nfe=self.nfe,
            runtime_seconds=runtime,
            metadata={
                "algorithm": "RAPO_EGH_ARO_POA",
                "hybrid_type": "Elite-guided Hybridization",
                "main_optimizer": "ARO",
                "elite_refiner": "POA_Phase_2_Defense",
                "population_size": self.population_size,
                "max_iterations": self.max_iterations,
                "top_k": self.top_k,
                "refinement_interval": self.refinement_interval,
                "local_iterations": self.local_iterations,
                "seed": self.seed,
                "max_evaluations": self.max_evaluations,
                "refinement_events": refinement_events,
                "iteration_trace": iteration_trace,
            },
        )

    # ------------------------- ARO operators -------------------------

    def _aro_energy(self, t: int) -> float:
        r = float(self.rng.uniform(1e-12, 1.0))
        return float(4.0 * (1.0 - t / self.max_iterations) * np.log(1.0 / r))

    def _aro_running_operator(self, dimension: int, t: int) -> Array:
        r2 = float(self.rng.random())
        r3 = float(self.rng.random())
        running_length = (np.e - np.exp(((t - 1.0) / self.max_iterations) ** 2)) * np.sin(2.0 * np.pi * r2)

        selected_count = max(1, int(np.ceil(r3 * dimension)))
        selected_dims = self.rng.choice(dimension, size=selected_count, replace=False)
        mask = np.zeros(dimension)
        mask[selected_dims] = 1.0
        return running_length * mask

    def _aro_detour(self, population: Array, i: int, t: int, lb: Array, ub: Array) -> Array:
        n, dim = population.shape
        candidates = [idx for idx in range(n) if idx != i]
        j = int(self.rng.choice(candidates))
        r1 = float(self.rng.random())
        perturb_flag = round(0.5 * (0.05 + r1))
        noise = self.rng.normal(0.0, 1.0, dim) * (ub - lb) * 0.001
        running_operator = self._aro_running_operator(dim, t)
        return population[j] + running_operator * (population[i] - population[j]) + perturb_flag * noise

    def _aro_random_hiding(self, xi: Array, t: int, lb: Array, ub: Array) -> Array:
        dim = xi.size
        random_dim = int(self.rng.integers(0, dim))
        mask = np.zeros(dim)
        mask[random_dim] = 1.0
        hiding_parameter = ((self.max_iterations - t + 1.0) / self.max_iterations) * float(self.rng.random())
        burrow = xi + hiding_parameter * mask * xi
        running_operator = self._aro_running_operator(dim, t)
        return xi + running_operator * (float(self.rng.random()) * burrow - xi)

    # ------------------------- POA local refiner -------------------------

    def _poa_refine_top_elites(
        self,
        objective_func: ObjectiveFunc,
        population: Array,
        fitness: Array,
        lb: Array,
        ub: Array,
        outer_iteration: int,
    ) -> int:
        """Refine top-k elites using POA Phase 2 local movement."""
        elite_indices = np.argsort(fitness)[: min(self.top_k, self.population_size)]
        refined_count = 0

        for idx in elite_indices:
            if self._should_stop():
                break

            current = population[idx].copy()
            current_fit = float(fitness[idx])

            for micro_t in range(1, self.local_iterations + 1):
                if self._should_stop():
                    break

                effective_t = max(1, outer_iteration + micro_t)
                candidate = self._poa_defense(current, effective_t, lb, ub)
                candidate = self._repair(candidate, lb, ub)
                candidate_fit = self._evaluate(objective_func, candidate)

                if candidate_fit <= current_fit:
                    current = candidate
                    current_fit = candidate_fit

            if current_fit <= fitness[idx]:
                population[idx] = current
                fitness[idx] = current_fit
                refined_count += 1

        return refined_count

    def _poa_defense(self, xi: Array, t: int, lb: Array, ub: Array) -> Array:
        t_safe = max(1, t)
        return xi + (1.0 - 2.0 * self.rng.random(xi.size)) * ((ub - lb) / t_safe)

    # ------------------------- Utilities -------------------------

    def _prepare_bounds(self, dimension: int) -> Tuple[Array, Array]:
        lb = np.full(dimension, self.lower_bound, dtype=float) if np.isscalar(self.lower_bound) else np.asarray(self.lower_bound, dtype=float)
        ub = np.full(dimension, self.upper_bound, dtype=float) if np.isscalar(self.upper_bound) else np.asarray(self.upper_bound, dtype=float)
        if lb.shape != (dimension,) or ub.shape != (dimension,):
            raise ValueError("lower_bound and upper_bound must be scalars or arrays with shape (dimension,)")
        if np.any(lb >= ub):
            raise ValueError("All lower bounds must be smaller than upper bounds")
        return lb, ub

    @staticmethod
    def _repair(x: Array, lb: Array, ub: Array) -> Array:
        return np.clip(x, lb, ub)

    def _evaluate(self, objective_func: ObjectiveFunc, x: Array) -> float:
        if self.max_evaluations is not None and self.nfe >= self.max_evaluations:
            self._budget_exhausted = True
            return float("inf")
        value = float(objective_func(np.asarray(x, dtype=float)))
        self.nfe += 1
        return value

    def _should_stop(self) -> bool:
        return bool(self._budget_exhausted or (self.max_evaluations is not None and self.nfe >= self.max_evaluations))


if __name__ == "__main__":
    def sphere(x: Array) -> float:
        return float(np.sum(x**2))

    optimizer = RAPOEliteGuidedAROPOA(
        population_size=50,
        max_iterations=100,
        lower_bound=-100,
        upper_bound=100,
        top_k=3,
        refinement_interval=10,
        local_iterations=5,
        seed=42,
    )
    result = optimizer.optimize(sphere, dimension=30)
    print("Algorithm:", result.metadata["algorithm"])
    print("Best fitness:", result.best_fitness)
    print("NFE:", result.nfe)
    print("Runtime seconds:", round(result.runtime_seconds, 6))
