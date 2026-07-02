"""
rapo_island_model.py

RAPO Island Model / Cooperative Co-evolution
--------------------------------------------
Hybrid mechanism:
    - Split the population into multiple islands.
    - Each island runs either ARO or POA.
    - Islands evolve independently within each iteration.
    - Every `migration_interval` iterations, elite solutions migrate in a ring topology.
    - Migrants replace the worst individuals in the destination island if they are better.

This file is intentionally standalone so it can be tested directly, then adapted into:
    metaheuristic_benchmark/algorithms/hybrids/rapo_island_model.py

Minimization is assumed.
Python: 3.11+
Dependency: numpy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import time

import numpy as np


Array = np.ndarray
ObjectiveFunction = Callable[[Array], float]


@dataclass
class OptimizationResult:
    """Simple standalone result object.

    Replace this dataclass with your framework's OptimizationResult if needed.
    """

    best_solution: Array
    best_fitness: float
    convergence_curve: List[float]
    nfe: int
    runtime_seconds: float
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class IslandState:
    """State of one island in the island-model hybrid."""

    island_id: int
    algorithm: str  # "ARO" or "POA"
    population: Array
    fitness: Array
    best_solution: Array
    best_fitness: float


class RAPOIslandModel:
    """RAPO Island Model / Cooperative Co-evolution.

    Mechanism:
        1. Initialize a shared population X.
        2. Split X into S islands.
        3. Assign ARO or POA mechanism to each island.
        4. Evolve islands independently.
        5. Every M iterations, migrate elite individuals in ring topology.

    Suggested registry name:
        register_algorithm("RAPO_Island_Model", RAPOIslandModel)

    Parameters
    ----------
    population_size:
        Total number of individuals over all islands.
    max_iterations:
        Maximum number of iterations T.
    lower_bound, upper_bound:
        Scalar or per-dimension search bounds.
    num_islands:
        Number of islands S.
    migration_interval:
        Migration interval M. If M <= 0, migration is disabled.
    elite_count:
        Number of elite migrants sent from each island per migration event.
    aro_island_ratio:
        Approximate ratio of islands assigned to ARO. The remaining islands use POA.
    assignment_mode:
        "alternating" gives ARO, POA, ARO, POA,... when possible.
        "block" assigns the first ARO islands then POA islands.
    seed:
        Random seed for reproducibility.
    max_evaluations:
        Optional evaluation budget stop condition for fair comparison.
    """

    def __init__(
        self,
        population_size: int = 50,
        max_iterations: int = 100,
        lower_bound: float | Sequence[float] = -100.0,
        upper_bound: float | Sequence[float] = 100.0,
        num_islands: int = 4,
        migration_interval: int = 10,
        elite_count: int = 1,
        aro_island_ratio: float = 0.5,
        assignment_mode: str = "alternating",
        seed: Optional[int] = None,
        max_evaluations: Optional[int] = None,
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        if num_islands < 2:
            raise ValueError("num_islands must be at least 2 for island-model hybridization.")
        if num_islands > population_size:
            raise ValueError("num_islands must not exceed population_size.")
        if elite_count < 1:
            raise ValueError("elite_count must be at least 1.")
        if not (0.0 <= aro_island_ratio <= 1.0):
            raise ValueError("aro_island_ratio must be in [0, 1].")
        if assignment_mode not in {"alternating", "block"}:
            raise ValueError("assignment_mode must be 'alternating' or 'block'.")

        self.population_size = int(population_size)
        self.max_iterations = int(max_iterations)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.num_islands = int(num_islands)
        self.migration_interval = int(migration_interval)
        self.elite_count = int(elite_count)
        self.aro_island_ratio = float(aro_island_ratio)
        self.assignment_mode = assignment_mode
        self.seed = seed
        self.max_evaluations = max_evaluations

        self.rng = np.random.default_rng(seed)
        self.nfe = 0

    def optimize(self, objective_func: ObjectiveFunction, dimension: int) -> OptimizationResult:
        """Run RAPO island model on a continuous minimization problem."""
        if dimension < 1:
            raise ValueError("dimension must be at least 1.")

        start_time = time.perf_counter()
        self.nfe = 0

        lb, ub = self._prepare_bounds(dimension)
        population = self.rng.uniform(lb, ub, size=(self.population_size, dimension))
        fitness = self._evaluate_population(objective_func, population)

        islands = self._create_islands(population, fitness)
        global_best_solution, global_best_fitness = self._get_global_best(islands)

        convergence_curve: List[float] = [float(global_best_fitness)]
        migration_events: List[Dict[str, object]] = []
        iteration_trace: List[Dict[str, object]] = []

        stop_reason = "max_iterations"
        for t in range(1, self.max_iterations + 1):
            if self._budget_exhausted():
                stop_reason = "max_evaluations"
                break

            for island in islands:
                if self._budget_exhausted():
                    stop_reason = "max_evaluations"
                    break

                if island.algorithm == "ARO":
                    self._evolve_aro_island(objective_func, island, lb, ub, t)
                elif island.algorithm == "POA":
                    self._evolve_poa_island(objective_func, island, lb, ub, t)
                else:
                    raise RuntimeError(f"Unsupported island algorithm: {island.algorithm}")

                self._refresh_island_best(island)

            global_best_solution, global_best_fitness = self._get_global_best(islands)

            migration_done = False
            if self._should_migrate(t):
                events = self._migrate_elites_ring(islands)
                migration_events.extend(events)
                migration_done = True

                for island in islands:
                    self._refresh_island_best(island)
                global_best_solution, global_best_fitness = self._get_global_best(islands)

            convergence_curve.append(float(global_best_fitness))
            iteration_trace.append(
                {
                    "iteration": t,
                    "global_best_fitness": float(global_best_fitness),
                    "nfe": self.nfe,
                    "migration_done": migration_done,
                    "island_best": [float(island.best_fitness) for island in islands],
                }
            )

        runtime_seconds = time.perf_counter() - start_time
        metadata = {
            "algorithm": "RAPO_Island_Model",
            "hybrid_type": "Island Model / Cooperative Co-evolution",
            "population_size": self.population_size,
            "max_iterations": self.max_iterations,
            "dimension": dimension,
            "num_islands": self.num_islands,
            "migration_interval": self.migration_interval,
            "elite_count": self.elite_count,
            "aro_island_ratio": self.aro_island_ratio,
            "assignment_mode": self.assignment_mode,
            "island_algorithms": [island.algorithm for island in islands],
            "island_sizes": [int(island.population.shape[0]) for island in islands],
            "migration_events": migration_events,
            "iteration_trace": iteration_trace,
            "seed": self.seed,
            "max_evaluations": self.max_evaluations,
            "stop_reason": stop_reason,
            "note": (
                "Standalone implementation. Replace _evaluate calls with evaluator.evaluate() "
                "when integrating into the full benchmark framework."
            ),
        }

        return OptimizationResult(
            best_solution=np.array(global_best_solution, dtype=float),
            best_fitness=float(global_best_fitness),
            convergence_curve=convergence_curve,
            nfe=self.nfe,
            runtime_seconds=float(runtime_seconds),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Island construction and migration
    # ------------------------------------------------------------------

    def _create_islands(self, population: Array, fitness: Array) -> List[IslandState]:
        algorithms = self._assign_island_algorithms()
        perm = self.rng.permutation(self.population_size)
        index_groups = np.array_split(perm, self.num_islands)

        islands: List[IslandState] = []
        for island_id, idx in enumerate(index_groups):
            pop_i = population[idx].copy()
            fit_i = fitness[idx].copy()
            best_idx = int(np.argmin(fit_i))
            islands.append(
                IslandState(
                    island_id=island_id,
                    algorithm=algorithms[island_id],
                    population=pop_i,
                    fitness=fit_i,
                    best_solution=pop_i[best_idx].copy(),
                    best_fitness=float(fit_i[best_idx]),
                )
            )
        return islands

    def _assign_island_algorithms(self) -> List[str]:
        aro_count = int(round(self.num_islands * self.aro_island_ratio))
        aro_count = min(max(aro_count, 0), self.num_islands)
        poa_count = self.num_islands - aro_count

        # Avoid all-one-algorithm assignment unless explicitly requested by ratio 0 or 1.
        if 0.0 < self.aro_island_ratio < 1.0:
            aro_count = min(max(aro_count, 1), self.num_islands - 1)
            poa_count = self.num_islands - aro_count

        if self.assignment_mode == "block":
            return ["ARO"] * aro_count + ["POA"] * poa_count

        # Alternating layout: ARO, POA, ARO, POA, then fill remaining.
        algorithms: List[str] = []
        a_left, p_left = aro_count, poa_count
        prefer_aro = True
        while len(algorithms) < self.num_islands:
            if prefer_aro and a_left > 0:
                algorithms.append("ARO")
                a_left -= 1
            elif (not prefer_aro) and p_left > 0:
                algorithms.append("POA")
                p_left -= 1
            elif a_left > 0:
                algorithms.append("ARO")
                a_left -= 1
            elif p_left > 0:
                algorithms.append("POA")
                p_left -= 1
            prefer_aro = not prefer_aro
        return algorithms

    def _should_migrate(self, iteration: int) -> bool:
        return self.migration_interval > 0 and iteration % self.migration_interval == 0

    def _migrate_elites_ring(self, islands: List[IslandState]) -> List[Dict[str, object]]:
        """Ring migration using best-to-worst replacement.

        Each island sends its top `elite_count` individuals to the next island.
        Destination island accepts a migrant only if the migrant is better than its
        currently selected worst individual.
        """
        packets: List[Tuple[int, int, Array, Array]] = []
        for source_pos, source in enumerate(islands):
            destination_pos = (source_pos + 1) % len(islands)
            k = min(self.elite_count, source.population.shape[0])
            elite_indices = np.argsort(source.fitness)[:k]
            packets.append(
                (
                    source.island_id,
                    islands[destination_pos].island_id,
                    source.population[elite_indices].copy(),
                    source.fitness[elite_indices].copy(),
                )
            )

        events: List[Dict[str, object]] = []
        for source_id, destination_id, migrants, migrant_fitness in packets:
            destination = next(island for island in islands if island.island_id == destination_id)
            worst_order = np.argsort(destination.fitness)[::-1]
            replacements = 0

            for local_migrant_idx, migrant in enumerate(migrants):
                if local_migrant_idx >= len(worst_order):
                    break
                worst_idx = int(worst_order[local_migrant_idx])
                migrant_fit = float(migrant_fitness[local_migrant_idx])
                if migrant_fit < float(destination.fitness[worst_idx]):
                    destination.population[worst_idx] = migrant.copy()
                    destination.fitness[worst_idx] = migrant_fit
                    replacements += 1

            events.append(
                {
                    "from_island": source_id,
                    "to_island": destination_id,
                    "migrants_sent": int(len(migrants)),
                    "accepted_replacements": int(replacements),
                }
            )
        return events

    # ------------------------------------------------------------------
    # ARO island evolution
    # ------------------------------------------------------------------

    def _evolve_aro_island(
        self,
        objective_func: ObjectiveFunction,
        island: IslandState,
        lb: Array,
        ub: Array,
        t: int,
    ) -> None:
        n = island.population.shape[0]
        for i in range(n):
            if self._budget_exhausted():
                return
            energy = self._aro_energy(t)
            if energy > 1.0:
                candidate = self._aro_detour_foraging(island.population, i, t)
            else:
                candidate = self._aro_random_hiding(island.population[i], t)
            self._greedy_update(objective_func, island, i, candidate, lb, ub)

    def _aro_energy(self, t: int) -> float:
        r = float(self.rng.random())
        r = max(r, 1e-12)
        return float(4.0 * (1.0 - t / self.max_iterations) * np.log(1.0 / r))

    def _aro_detour_foraging(self, population: Array, i: int, t: int) -> Array:
        n, d = population.shape
        xi = population[i]
        if n > 1:
            candidates = [idx for idx in range(n) if idx != i]
            j = int(self.rng.choice(candidates))
            xj = population[j]
        else:
            xj = xi.copy()

        running_operator = self._aro_running_operator(d, t)
        perturb_flag = round(0.5 * (0.05 + float(self.rng.random())))
        gaussian_noise = self.rng.normal(0.0, 1.0, size=d)
        return xj + running_operator * (xi - xj) + perturb_flag * gaussian_noise

    def _aro_random_hiding(self, xi: Array, t: int) -> Array:
        d = xi.shape[0]
        random_dim = int(self.rng.integers(0, d))
        gr = np.zeros(d, dtype=float)
        gr[random_dim] = 1.0

        hiding_parameter = ((self.max_iterations - t + 1) / self.max_iterations) * float(self.rng.random())
        burrow = xi + hiding_parameter * gr * xi
        running_operator = self._aro_running_operator(d, t)
        return xi + running_operator * (float(self.rng.random()) * burrow - xi)

    def _aro_running_operator(self, dimension: int, t: int) -> Array:
        # Running length used by ARO, with a random sparse coordinate mask.
        r2 = float(self.rng.random())
        length = (np.e - np.exp(((t - 1) / self.max_iterations) ** 2.0)) * np.sin(2.0 * np.pi * r2)

        selected_count = max(1, int(np.ceil(float(self.rng.random()) * dimension)))
        selected_dims = self.rng.choice(dimension, size=selected_count, replace=False)
        mask = np.zeros(dimension, dtype=float)
        mask[selected_dims] = 1.0
        return length * mask

    # ------------------------------------------------------------------
    # POA island evolution
    # ------------------------------------------------------------------

    def _evolve_poa_island(
        self,
        objective_func: ObjectiveFunction,
        island: IslandState,
        lb: Array,
        ub: Array,
        t: int,
    ) -> None:
        n = island.population.shape[0]
        for i in range(n):
            if self._budget_exhausted():
                return

            # Phase 1: Predator Attack towards Pufferfish - exploration.
            candidate_phase_1 = self._poa_predator_attack(island.population, island.fitness, i)
            if candidate_phase_1 is not None:
                self._greedy_update(objective_func, island, i, candidate_phase_1, lb, ub)

            if self._budget_exhausted():
                return

            # Phase 2: Defense Mechanism - exploitation.
            candidate_phase_2 = self._poa_defense_mechanism(island.population[i], lb, ub, t)
            self._greedy_update(objective_func, island, i, candidate_phase_2, lb, ub)

    def _poa_predator_attack(self, population: Array, fitness: Array, i: int) -> Optional[Array]:
        better_indices = np.where(fitness < fitness[i])[0]
        better_indices = better_indices[better_indices != i]
        if better_indices.size == 0:
            return None

        selected_idx = int(self.rng.choice(better_indices))
        selected_pufferfish = population[selected_idx]
        xi = population[i]
        r = self.rng.random(size=xi.shape[0])
        integer_factor = self.rng.integers(1, 3, size=xi.shape[0])  # values 1 or 2
        return xi + r * (selected_pufferfish - integer_factor * xi)

    def _poa_defense_mechanism(self, xi: Array, lb: Array, ub: Array, t: int) -> Array:
        r = self.rng.random(size=xi.shape[0])
        return xi + (1.0 - 2.0 * r) * ((ub - lb) / max(t, 1))

    # ------------------------------------------------------------------
    # Evaluation and utility
    # ------------------------------------------------------------------

    def _greedy_update(
        self,
        objective_func: ObjectiveFunction,
        island: IslandState,
        i: int,
        candidate: Array,
        lb: Array,
        ub: Array,
    ) -> None:
        candidate = self._repair_bounds(candidate, lb, ub)
        candidate_fitness = self._evaluate(objective_func, candidate)
        if candidate_fitness <= float(island.fitness[i]):
            island.population[i] = candidate
            island.fitness[i] = candidate_fitness

    def _refresh_island_best(self, island: IslandState) -> None:
        best_idx = int(np.argmin(island.fitness))
        island.best_solution = island.population[best_idx].copy()
        island.best_fitness = float(island.fitness[best_idx])

    def _get_global_best(self, islands: List[IslandState]) -> Tuple[Array, float]:
        best_island = min(islands, key=lambda isl: isl.best_fitness)
        return best_island.best_solution.copy(), float(best_island.best_fitness)

    def _evaluate(self, objective_func: ObjectiveFunction, x: Array) -> float:
        if self._budget_exhausted():
            # Return +inf if budget is exhausted before a candidate can be evaluated.
            return float("inf")
        value = float(objective_func(np.asarray(x, dtype=float)))
        self.nfe += 1
        return value

    def _evaluate_population(self, objective_func: ObjectiveFunction, population: Array) -> Array:
        values = []
        for individual in population:
            values.append(self._evaluate(objective_func, individual))
        return np.asarray(values, dtype=float)

    def _budget_exhausted(self) -> bool:
        return self.max_evaluations is not None and self.nfe >= self.max_evaluations

    def _prepare_bounds(self, dimension: int) -> Tuple[Array, Array]:
        lb = np.asarray(self.lower_bound, dtype=float)
        ub = np.asarray(self.upper_bound, dtype=float)

        if lb.ndim == 0:
            lb = np.full(dimension, float(lb))
        if ub.ndim == 0:
            ub = np.full(dimension, float(ub))

        if lb.shape != (dimension,) or ub.shape != (dimension,):
            raise ValueError("lower_bound and upper_bound must be scalars or arrays of shape (dimension,).")
        if np.any(lb >= ub):
            raise ValueError("Each lower_bound value must be less than upper_bound.")
        return lb, ub

    @staticmethod
    def _repair_bounds(x: Array, lb: Array, ub: Array) -> Array:
        return np.clip(np.asarray(x, dtype=float), lb, ub)


# ----------------------------------------------------------------------
# Demo run
# ----------------------------------------------------------------------


def sphere(x: Array) -> float:
    """Sphere benchmark. Global minimum: f(0,...,0) = 0."""
    return float(np.sum(x ** 2))


if __name__ == "__main__":
    optimizer = RAPOIslandModel(
        population_size=60,
        max_iterations=100,
        lower_bound=-100.0,
        upper_bound=100.0,
        num_islands=4,
        migration_interval=10,
        elite_count=1,
        aro_island_ratio=0.5,
        assignment_mode="alternating",
        seed=42,
        max_evaluations=None,
    )

    result = optimizer.optimize(objective_func=sphere, dimension=30)

    print("Algorithm:", result.metadata["algorithm"])
    print("Best fitness:", result.best_fitness)
    print("Best solution shape:", result.best_solution.shape)
    print("NFE:", result.nfe)
    print("Runtime seconds:", round(result.runtime_seconds, 6))
    print("Island algorithms:", result.metadata["island_algorithms"])
    print("Migration events:", len(result.metadata["migration_events"]))
