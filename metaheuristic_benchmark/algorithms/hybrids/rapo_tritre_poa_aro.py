"""
RAPO-STH-POA-ARO: Stagnation-triggered Hybridization
=======================================================

Cơ chế lai:
- POA chạy chính trên toàn bộ quần thể.
- Nếu best fitness không cải thiện đáng kể sau `stagnation_patience` vòng lặp,
  kích hoạt ARO escape để làm mới / khai thác nhóm cá thể xấu nhất.
- ARO escape dùng Energy Shrink để chọn Detour Foraging hoặc Random Hiding
  trong `escape_iterations` vòng phụ.

Bài toán: minimization.
Python: 3.11+
Phụ thuộc: numpy

Gợi ý đặt file:
metaheuristic_benchmark/algorithms/hybrids/rapo_stagnation_triggered_poa_aro.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import math
import time
import numpy as np


ObjectiveFunc = Callable[[np.ndarray], float]


@dataclass
class OptimizationResult:
    best_solution: np.ndarray
    best_fitness: float
    convergence_curve: List[float]
    nfe: int
    runtime_seconds: float
    metadata: Dict[str, object] = field(default_factory=dict)


class RAPOStagnationTriggeredPOAARO:
    """
    POA-main + ARO-escape khi trì trệ.

    Ý nghĩa tham số chính:
    - population_size: số cá thể N.
    - max_iterations: số vòng lặp chính T.
    - lower_bound, upper_bound: biên dưới / biên trên của không gian tìm kiếm.
    - stagnation_patience: số vòng không cải thiện đáng kể trước khi kích hoạt ARO.
    - improvement_epsilon: ngưỡng cải thiện đáng kể, dùng cho điều kiện:
        new_best < old_best - improvement_epsilon
    - escape_iterations: số vòng phụ L cho ARO escape.
    - escape_ratio: tỷ lệ cá thể xấu nhất được ARO xử lý, ví dụ 0.30 = 30%.
    - max_evaluations: ngân sách đánh giá hàm mục tiêu, dùng để so sánh công bằng.
    """

    name = "RAPO_STH_POA_ARO"

    def __init__(
        self,
        population_size: int = 50,
        max_iterations: int = 100,
        lower_bound: float | np.ndarray = -100.0,
        upper_bound: float | np.ndarray = 100.0,
        stagnation_patience: int = 5,
        improvement_epsilon: float = 1e-6,
        escape_iterations: int = 5,
        escape_ratio: float = 0.30,
        max_evaluations: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size phải >= 2.")
        if max_iterations < 1:
            raise ValueError("max_iterations phải >= 1.")
        if stagnation_patience < 1:
            raise ValueError("stagnation_patience phải >= 1.")
        if escape_iterations < 1:
            raise ValueError("escape_iterations phải >= 1.")
        if not (0.0 < escape_ratio <= 1.0):
            raise ValueError("escape_ratio phải nằm trong (0, 1].")

        self.population_size = int(population_size)
        self.max_iterations = int(max_iterations)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.stagnation_patience = int(stagnation_patience)
        self.improvement_epsilon = float(improvement_epsilon)
        self.escape_iterations = int(escape_iterations)
        self.escape_ratio = float(escape_ratio)
        self.max_evaluations = max_evaluations
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.nfe = 0
        self._budget_exhausted = False

    def optimize(self, objective_func: ObjectiveFunc, dimension: int) -> OptimizationResult:
        if dimension < 1:
            raise ValueError("dimension phải >= 1.")

        start_time = time.perf_counter()
        lb, ub = self._make_bounds(dimension)

        self.nfe = 0
        self._budget_exhausted = False

        population = self.rng.uniform(lb, ub, size=(self.population_size, dimension))
        fitness = np.array([self._evaluate(objective_func, x) for x in population], dtype=float)

        best_idx = int(np.argmin(fitness))
        best_solution = population[best_idx].copy()
        best_fitness = float(fitness[best_idx])

        convergence_curve: List[float] = [best_fitness]
        trigger_events: List[Dict[str, object]] = []
        iteration_trace: List[Dict[str, object]] = []

        stagnation_count = 0

        for t in range(1, self.max_iterations + 1):
            if self._budget_exhausted:
                break

            old_best = best_fitness
            poa_updates = 0

            # 1) Chạy POA chính: Phase 1 + Phase 2 cho từng cá thể.
            for i in range(self.population_size):
                if self._budget_exhausted:
                    break

                # Phase 1: Predator Attack.
                candidate1 = self._poa_attack_candidate(population, fitness, i)
                if candidate1 is not None:
                    candidate1 = self._repair(candidate1, lb, ub)
                    fit1 = self._evaluate(objective_func, candidate1)
                    if fit1 <= fitness[i]:
                        population[i] = candidate1
                        fitness[i] = fit1
                        poa_updates += 1

                # Phase 2: Defense Mechanism.
                if self._budget_exhausted:
                    break
                candidate2 = self._poa_defense_candidate(population, i, t, lb, ub)
                candidate2 = self._repair(candidate2, lb, ub)
                fit2 = self._evaluate(objective_func, candidate2)
                if fit2 <= fitness[i]:
                    population[i] = candidate2
                    fitness[i] = fit2
                    poa_updates += 1

                if fitness[i] < best_fitness:
                    best_fitness = float(fitness[i])
                    best_solution = population[i].copy()

            # 2) Kiểm tra trì trệ.
            if best_fitness < old_best - self.improvement_epsilon:
                stagnation_count = 0
                improved = True
            else:
                stagnation_count += 1
                improved = False

            # 3) Nếu trì trệ đủ lâu, kích hoạt ARO escape.
            aro_escape_updates = 0
            triggered = False
            if stagnation_count >= self.stagnation_patience and not self._budget_exhausted:
                triggered = True
                selected_indices = self._worst_indices(fitness, self.escape_ratio)
                before_escape_best = best_fitness

                for micro_iter in range(1, self.escape_iterations + 1):
                    if self._budget_exhausted:
                        break
                    for idx in selected_indices:
                        if self._budget_exhausted:
                            break

                        candidate = self._aro_candidate(population, idx, micro_iter, lb, ub)
                        candidate = self._repair(candidate, lb, ub)
                        candidate_fit = self._evaluate(objective_func, candidate)
                        if candidate_fit <= fitness[idx]:
                            population[idx] = candidate
                            fitness[idx] = candidate_fit
                            aro_escape_updates += 1
                            if candidate_fit < best_fitness:
                                best_fitness = float(candidate_fit)
                                best_solution = candidate.copy()

                trigger_events.append(
                    {
                        "iteration": t,
                        "trigger": "ARO_escape",
                        "selected_count": int(len(selected_indices)),
                        "escape_iterations": self.escape_iterations,
                        "best_before_escape": float(before_escape_best),
                        "best_after_escape": float(best_fitness),
                        "updates": int(aro_escape_updates),
                    }
                )
                stagnation_count = 0

            convergence_curve.append(best_fitness)
            iteration_trace.append(
                {
                    "iteration": t,
                    "best_fitness": float(best_fitness),
                    "improved": improved,
                    "stagnation_count": int(stagnation_count),
                    "poa_updates": int(poa_updates),
                    "aro_escape_triggered": triggered,
                    "aro_escape_updates": int(aro_escape_updates),
                    "nfe": int(self.nfe),
                }
            )

        runtime = time.perf_counter() - start_time
        return OptimizationResult(
            best_solution=best_solution,
            best_fitness=best_fitness,
            convergence_curve=convergence_curve,
            nfe=self.nfe,
            runtime_seconds=runtime,
            metadata={
                "algorithm": self.name,
                "population_size": self.population_size,
                "max_iterations": self.max_iterations,
                "stagnation_patience": self.stagnation_patience,
                "improvement_epsilon": self.improvement_epsilon,
                "escape_iterations": self.escape_iterations,
                "escape_ratio": self.escape_ratio,
                "max_evaluations": self.max_evaluations,
                "seed": self.seed,
                "trigger_events": trigger_events,
                "iteration_trace": iteration_trace,
                "budget_exhausted": self._budget_exhausted,
            },
        )

    # ------------------------------------------------------------------
    # POA operators
    # ------------------------------------------------------------------
    def _poa_attack_candidate(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        i: int,
    ) -> Optional[np.ndarray]:
        better_indices = np.where(fitness < fitness[i])[0]
        better_indices = better_indices[better_indices != i]
        if better_indices.size == 0:
            return None

        sp_idx = int(self.rng.choice(better_indices))
        xi = population[i]
        sp = population[sp_idx]
        r = self.rng.random(size=xi.shape)
        integer_factor = self.rng.integers(1, 3, size=xi.shape)  # 1 hoặc 2
        return xi + r * (sp - integer_factor * xi)

    def _poa_defense_candidate(
        self,
        population: np.ndarray,
        i: int,
        t: int,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> np.ndarray:
        xi = population[i]
        r = self.rng.random(size=xi.shape)
        return xi + (1.0 - 2.0 * r) * ((ub - lb) / max(float(t), 1.0))

    # ------------------------------------------------------------------
    # ARO operators
    # ------------------------------------------------------------------
    def _aro_candidate(
        self,
        population: np.ndarray,
        i: int,
        t: int,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> np.ndarray:
        """Energy Shrink quyết định ARO Detour Foraging hoặc Random Hiding."""
        energy = self._energy_factor(t)
        if energy > 1.0:
            return self._aro_detour_foraging(population, i)
        return self._aro_random_hiding(population, i, t)

    def _aro_detour_foraging(self, population: np.ndarray, i: int) -> np.ndarray:
        n, d = population.shape
        j = int(self.rng.integers(0, n - 1))
        if j >= i:
            j += 1

        r1 = self.rng.random()
        r2 = self.rng.random()
        r3 = self.rng.random()

        length = (math.e - math.exp(0.0)) * math.sin(2.0 * math.pi * r2)
        c = np.zeros(d)
        selected_count = max(1, int(math.ceil(r3 * d)))
        selected_dims = self.rng.permutation(d)[:selected_count]
        c[selected_dims] = 1.0
        running_operator = length * c

        perturb = round(0.5 * (0.05 + r1)) * self.rng.normal(0.0, 1.0, size=d)
        return population[j] + running_operator * (population[i] - population[j]) + perturb

    def _aro_random_hiding(self, population: np.ndarray, i: int, t: int) -> np.ndarray:
        _, d = population.shape
        xi = population[i]
        selected_dim = int(self.rng.integers(0, d))
        mask = np.zeros(d)
        mask[selected_dim] = 1.0

        h = ((self.max_iterations - t + 1.0) / self.max_iterations) * self.rng.random()
        burrow = xi + h * mask * xi

        r2 = self.rng.random()
        length = (math.e - math.exp(((t - 1.0) / self.max_iterations) ** 2)) * math.sin(2.0 * math.pi * r2)
        running_operator = length * mask
        return xi + running_operator * (self.rng.random() * burrow - xi)

    def _energy_factor(self, t: int) -> float:
        r = max(float(self.rng.random()), 1e-12)
        return 4.0 * (1.0 - float(t) / float(self.max_iterations)) * math.log(1.0 / r)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _make_bounds(self, dimension: int) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.full(dimension, self.lower_bound, dtype=float) if np.isscalar(self.lower_bound) else np.asarray(self.lower_bound, dtype=float)
        ub = np.full(dimension, self.upper_bound, dtype=float) if np.isscalar(self.upper_bound) else np.asarray(self.upper_bound, dtype=float)
        if lb.shape != (dimension,) or ub.shape != (dimension,):
            raise ValueError("lower_bound và upper_bound phải là scalar hoặc vector có độ dài bằng dimension.")
        if np.any(lb >= ub):
            raise ValueError("Mỗi lower_bound phải nhỏ hơn upper_bound.")
        return lb, ub

    def _repair(self, x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        return np.clip(x, lb, ub)

    def _evaluate(self, objective_func: ObjectiveFunc, x: np.ndarray) -> float:
        if self.max_evaluations is not None and self.nfe >= self.max_evaluations:
            self._budget_exhausted = True
            return float("inf")
        value = float(objective_func(np.asarray(x, dtype=float)))
        self.nfe += 1
        if self.max_evaluations is not None and self.nfe >= self.max_evaluations:
            self._budget_exhausted = True
        return value

    def _worst_indices(self, fitness: np.ndarray, ratio: float) -> np.ndarray:
        count = max(1, int(math.ceil(len(fitness) * ratio)))
        return np.argsort(fitness)[-count:]


if __name__ == "__main__":
    def sphere(x: np.ndarray) -> float:
        return float(np.sum(x ** 2))

    optimizer = RAPOStagnationTriggeredPOAARO(
        population_size=40,
        max_iterations=100,
        lower_bound=-100,
        upper_bound=100,
        stagnation_patience=5,
        improvement_epsilon=1e-6,
        escape_iterations=5,
        escape_ratio=0.30,
        seed=42,
    )
    result = optimizer.optimize(sphere, dimension=30)
    print("Algorithm:", optimizer.name)
    print("Best fitness:", result.best_fitness)
    print("NFE:", result.nfe)
    print("Runtime:", result.runtime_seconds)
    print("Trigger events:", result.metadata["trigger_events"][:3])
