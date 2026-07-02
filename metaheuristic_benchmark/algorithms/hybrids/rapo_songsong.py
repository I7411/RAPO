"""
rapo_parallel_hybrid.py

RAPO Full Parallel Hybrid / Teamwork (ARO-POA)
================================================

Cơ chế lai:
    - ARO và POA cùng sinh nghiệm ứng viên cho cùng một cá thể Xi.
    - Nhánh ARO dùng Energy Shrink để chọn:
        + A(t) > 1  : ARO Detour Foraging  (exploration)
        + A(t) <= 1 : ARO Random Hiding    (exploitation)
    - Nhánh POA sinh nghiệm theo cơ chế Pufferfish POA:
        + Phase 1: Predator Attack
        + Phase 2: Defense Mechanism
    - Sau đó chọn nghiệm tốt nhất giữa:
        Xi hiện tại, nghiệm ứng viên ARO, nghiệm ứng viên POA.
    - Cập nhật X_best toàn cục theo greedy selection cho bài toán minimization.

Lưu ý:
    Đây là bản standalone để dễ kiểm thử. Khi tích hợp vào benchmark framework thật,
    có thể thay self._evaluate(...) bằng evaluator.evaluate(...), và thay
    OptimizationResult bằng class result chung của project.

Python: 3.11+
Dependency: numpy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


Array = np.ndarray
ObjectiveFunc = Callable[[Array], float]


@dataclass
class OptimizationResult:
    """Kết quả tối ưu hóa tối thiểu đủ dùng cho benchmark framework."""

    algorithm_name: str
    best_solution: Array
    best_fitness: float
    convergence_curve: List[float]
    nfe: int
    runtime_seconds: float
    metadata: Dict[str, object] = field(default_factory=dict)


class RAPOParallelHybrid:
    """
    RAPO Full Parallel Hybrid / Teamwork.

    Mỗi cá thể Xi được cập nhật bằng cơ chế teamwork:
        1. Nhánh ARO sinh một nghiệm ứng viên V_aro.
        2. Nhánh POA sinh một hoặc nhiều nghiệm ứng viên V_poa.
        3. Chọn nghiệm tốt nhất trong {Xi, V_aro, V_poa}.

    Parameters
    ----------
    population_size:
        Số cá thể N.
    max_iterations:
        Số vòng lặp tối đa T.
    lower_bound, upper_bound:
        Cận dưới và cận trên. Có thể là scalar hoặc vector độ dài dimension.
    poa_mode:
        Cách sinh nghiệm ứng viên của nhánh POA.
        - "both_best": sinh cả POA Phase 1 và POA Phase 2, chọn nghiệm POA tốt hơn.
        - "energy"   : nếu A(t) > 1 dùng POA Attack, ngược lại dùng POA Defense.
        - "attack"   : chỉ dùng POA Phase 1.
        - "defense"  : chỉ dùng POA Phase 2.
    max_evaluations:
        Ngân sách đánh giá hàm mục tiêu. Nên dùng khi so sánh công bằng vì
        nhánh parallel có thể dùng nhiều NFE hơn thuật toán đơn.
    seed:
        Random seed để tái lập kết quả.
    """

    VALID_POA_MODES = {"both_best", "energy", "attack", "defense"}

    def __init__(
        self,
        population_size: int = 50,
        max_iterations: int = 100,
        lower_bound: float | Sequence[float] | Array = -100.0,
        upper_bound: float | Sequence[float] | Array = 100.0,
        poa_mode: str = "both_best",
        max_evaluations: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size phải >= 2.")
        if max_iterations < 1:
            raise ValueError("max_iterations phải >= 1.")
        if poa_mode not in self.VALID_POA_MODES:
            raise ValueError(f"poa_mode phải thuộc {sorted(self.VALID_POA_MODES)}.")
        if max_evaluations is not None and max_evaluations < population_size:
            raise ValueError("max_evaluations nên >= population_size để đủ đánh giá quần thể ban đầu.")

        self.population_size = int(population_size)
        self.max_iterations = int(max_iterations)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.poa_mode = poa_mode
        self.max_evaluations = max_evaluations
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.nfe = 0
        self._lb: Array | None = None
        self._ub: Array | None = None

    def optimize(self, objective_func: ObjectiveFunc, dimension: int) -> OptimizationResult:
        """Chạy RAPO Parallel Hybrid trên một objective function liên tục."""
        if dimension < 1:
            raise ValueError("dimension phải >= 1.")

        start_time = perf_counter()
        self.nfe = 0
        self._lb, self._ub = self._normalize_bounds(dimension)

        population = self.rng.uniform(
            low=self._lb,
            high=self._ub,
            size=(self.population_size, dimension),
        )
        fitness = np.array([self._evaluate(objective_func, x) for x in population], dtype=float)

        best_idx = int(np.argmin(fitness))
        best_solution = population[best_idx].copy()
        best_fitness = float(fitness[best_idx])

        convergence_curve: List[float] = [best_fitness]
        iteration_trace: List[Dict[str, object]] = []
        candidate_usage = {
            "ARO_DETOUR": 0,
            "ARO_HIDING": 0,
            "POA_ATTACK": 0,
            "POA_DEFENSE": 0,
        }
        winner_count = {
            "CURRENT": 0,
            "ARO_DETOUR": 0,
            "ARO_HIDING": 0,
            "POA_ATTACK": 0,
            "POA_DEFENSE": 0,
        }

        stop_reason = "max_iterations"

        for t in range(1, self.max_iterations + 1):
            if self._budget_exhausted():
                stop_reason = "max_evaluations"
                break

            iteration_winners = {key: 0 for key in winner_count}
            iteration_best_before = best_fitness

            for i in range(self.population_size):
                if self._budget_exhausted():
                    stop_reason = "max_evaluations"
                    break

                # Danh sách ứng viên gồm nghiệm hiện tại và các nghiệm do ARO/POA sinh ra.
                candidates: List[Tuple[str, Array, float]] = [
                    ("CURRENT", population[i].copy(), float(fitness[i]))
                ]

                energy = self._energy_factor(t)

                # ------------------------
                # Nhánh ARO chạy song song.
                # ------------------------
                aro_name, aro_candidate = self._aro_team_candidate(
                    population=population,
                    index=i,
                    iteration=t,
                    dimension=dimension,
                    energy=energy,
                )
                candidate_usage[aro_name] += 1
                self._append_evaluated_candidate(
                    objective_func=objective_func,
                    candidates=candidates,
                    name=aro_name,
                    candidate=aro_candidate,
                )

                # ------------------------
                # Nhánh POA chạy song song.
                # ------------------------
                poa_candidates = self._poa_team_candidates(
                    population=population,
                    fitness=fitness,
                    index=i,
                    iteration=t,
                    best_solution=best_solution,
                    energy=energy,
                )
                for poa_name, poa_candidate in poa_candidates:
                    if self._budget_exhausted():
                        stop_reason = "max_evaluations"
                        break
                    candidate_usage[poa_name] += 1
                    self._append_evaluated_candidate(
                        objective_func=objective_func,
                        candidates=candidates,
                        name=poa_name,
                        candidate=poa_candidate,
                    )

                # Chọn nghiệm tốt nhất giữa Xi, V_aro, V_poa theo minimization.
                selected_name, selected_solution, selected_fitness = min(
                    candidates,
                    key=lambda item: item[2],
                )
                winner_count[selected_name] += 1
                iteration_winners[selected_name] += 1

                if selected_fitness <= fitness[i]:
                    population[i] = selected_solution
                    fitness[i] = selected_fitness

                    if selected_fitness <= best_fitness:
                        best_fitness = float(selected_fitness)
                        best_solution = selected_solution.copy()

            convergence_curve.append(best_fitness)
            iteration_trace.append(
                {
                    "iteration": t,
                    "best_fitness_before": iteration_best_before,
                    "best_fitness_after": best_fitness,
                    "nfe": self.nfe,
                    "winner_count": iteration_winners,
                }
            )

            if stop_reason == "max_evaluations":
                break

        runtime_seconds = perf_counter() - start_time
        return OptimizationResult(
            algorithm_name="RAPO_Parallel_Hybrid",
            best_solution=best_solution,
            best_fitness=best_fitness,
            convergence_curve=convergence_curve,
            nfe=self.nfe,
            runtime_seconds=runtime_seconds,
            metadata={
                "hybrid_type": "Parallel Hybrid / Teamwork",
                "selection_rule": "best_of_current_aro_poa_candidates",
                "poa_mode": self.poa_mode,
                "population_size": self.population_size,
                "max_iterations": self.max_iterations,
                "max_evaluations": self.max_evaluations,
                "seed": self.seed,
                "candidate_usage": candidate_usage,
                "winner_count": winner_count,
                "iteration_trace": iteration_trace,
                "stop_reason": stop_reason,
            },
        )

    # ------------------------------------------------------------------
    # Toán tử ARO
    # ------------------------------------------------------------------
    def _aro_team_candidate(
        self,
        population: Array,
        index: int,
        iteration: int,
        dimension: int,
        energy: float,
    ) -> Tuple[str, Array]:
        """Sinh nghiệm ứng viên từ nhánh ARO theo Energy Shrink."""
        if energy > 1.0:
            return "ARO_DETOUR", self._aro_detour_foraging(population, index, iteration, dimension)
        return "ARO_HIDING", self._aro_random_hiding(population, index, iteration, dimension)

    def _aro_detour_foraging(
        self,
        population: Array,
        index: int,
        iteration: int,
        dimension: int,
    ) -> Array:
        """ARO Detour Foraging - pha khám phá."""
        n = population.shape[0]
        xi = population[index]

        j = int(self.rng.integers(0, n - 1))
        if j >= index:
            j += 1
        xj = population[j]

        r1 = float(self.rng.random())
        r2 = float(self.rng.random())
        r3 = float(self.rng.random())

        time_ratio = (iteration - 1) / max(1, self.max_iterations)
        length = (np.e - np.exp(time_ratio**2)) * np.sin(2.0 * np.pi * r2)

        c = np.zeros(dimension, dtype=float)
        selected_count = max(1, int(np.ceil(r3 * dimension)))
        selected_dims = self.rng.choice(dimension, size=selected_count, replace=False)
        c[selected_dims] = 1.0

        running_operator = length * c
        perturb_flag = round(0.5 * (0.05 + r1))
        gaussian_noise = self.rng.normal(0.0, 1.0, size=dimension)

        return xj + running_operator * (xi - xj) + perturb_flag * gaussian_noise

    def _aro_random_hiding(
        self,
        population: Array,
        index: int,
        iteration: int,
        dimension: int,
    ) -> Array:
        """ARO Random Hiding - pha khai thác."""
        xi = population[index]
        random_dim = int(self.rng.integers(0, dimension))

        gr = np.zeros(dimension, dtype=float)
        gr[random_dim] = 1.0

        hiding_factor = ((self.max_iterations - iteration + 1) / self.max_iterations) * float(
            self.rng.random()
        )
        random_burrow = xi + hiding_factor * gr * xi

        r4 = float(self.rng.random())
        r5 = float(self.rng.random())
        time_ratio = (iteration - 1) / max(1, self.max_iterations)
        length = (np.e - np.exp(time_ratio**2)) * np.sin(2.0 * np.pi * r5)

        c = np.zeros(dimension, dtype=float)
        selected_count = max(1, int(np.ceil(float(self.rng.random()) * dimension)))
        selected_dims = self.rng.choice(dimension, size=selected_count, replace=False)
        c[selected_dims] = 1.0

        running_operator = length * c
        return xi + running_operator * (r4 * random_burrow - xi)

    # ------------------------------------------------------------------
    # Toán tử POA-Pufferfish
    # ------------------------------------------------------------------
    def _poa_team_candidates(
        self,
        population: Array,
        fitness: Array,
        index: int,
        iteration: int,
        best_solution: Array,
        energy: float,
    ) -> List[Tuple[str, Array]]:
        """Sinh nghiệm ứng viên từ nhánh POA theo poa_mode."""
        if self.poa_mode == "both_best":
            return [
                ("POA_ATTACK", self._poa_predator_attack(population, fitness, index, best_solution)),
                ("POA_DEFENSE", self._poa_defense_mechanism(population[index], iteration)),
            ]
        if self.poa_mode == "energy":
            if energy > 1.0:
                return [("POA_ATTACK", self._poa_predator_attack(population, fitness, index, best_solution))]
            return [("POA_DEFENSE", self._poa_defense_mechanism(population[index], iteration))]
        if self.poa_mode == "attack":
            return [("POA_ATTACK", self._poa_predator_attack(population, fitness, index, best_solution))]
        if self.poa_mode == "defense":
            return [("POA_DEFENSE", self._poa_defense_mechanism(population[index], iteration))]
        raise RuntimeError("poa_mode không hợp lệ.")

    def _poa_predator_attack(
        self,
        population: Array,
        fitness: Array,
        index: int,
        best_solution: Array,
    ) -> Array:
        """POA Phase 1 - Predator Attack towards Pufferfish."""
        xi = population[index]
        better_indices = np.where(fitness < fitness[index])[0]
        better_indices = better_indices[better_indices != index]

        if better_indices.size > 0:
            selected_index = int(self.rng.choice(better_indices))
            selected_pufferfish = population[selected_index]
        else:
            # Nếu Xi đang là cá thể tốt nhất, dùng X_best làm fallback.
            selected_pufferfish = best_solution

        r = self.rng.random(size=xi.shape[0])
        integer_factor = self.rng.integers(1, 3, size=xi.shape[0])  # I thuộc {1, 2}
        return xi + r * (selected_pufferfish - integer_factor * xi)

    def _poa_defense_mechanism(self, xi: Array, iteration: int) -> Array:
        """POA Phase 2 - Defense Mechanism of Pufferfish."""
        assert self._lb is not None and self._ub is not None
        r = self.rng.random(size=xi.shape[0])
        step = (1.0 - 2.0 * r) * ((self._ub - self._lb) / max(1, iteration))
        return xi + step

    # ------------------------------------------------------------------
    # Hạ tầng dùng chung
    # ------------------------------------------------------------------
    def _append_evaluated_candidate(
        self,
        objective_func: ObjectiveFunc,
        candidates: List[Tuple[str, Array, float]],
        name: str,
        candidate: Array,
    ) -> None:
        """Repair, evaluate và thêm ứng viên nếu còn budget."""
        if self._budget_exhausted():
            return
        repaired = self._repair(candidate)
        fit = self._evaluate(objective_func, repaired)
        candidates.append((name, repaired, fit))

    def _energy_factor(self, iteration: int) -> float:
        """Energy Shrink của ARO: A(t) = 4(1 - t/T) ln(1/r)."""
        r = max(float(self.rng.random()), 1e-12)
        return 4.0 * (1.0 - iteration / self.max_iterations) * np.log(1.0 / r)

    def _evaluate(self, objective_func: ObjectiveFunc, solution: Array) -> float:
        if self._budget_exhausted():
            raise RuntimeError("Đã hết max_evaluations nhưng vẫn gọi evaluate.")
        self.nfe += 1
        value = float(objective_func(solution))
        if not np.isfinite(value):
            return float("inf")
        return value

    def _budget_exhausted(self) -> bool:
        return self.max_evaluations is not None and self.nfe >= self.max_evaluations

    def _normalize_bounds(self, dimension: int) -> Tuple[Array, Array]:
        lb = np.asarray(self.lower_bound, dtype=float)
        ub = np.asarray(self.upper_bound, dtype=float)

        if lb.ndim == 0:
            lb = np.full(dimension, float(lb), dtype=float)
        if ub.ndim == 0:
            ub = np.full(dimension, float(ub), dtype=float)

        if lb.shape != (dimension,) or ub.shape != (dimension,):
            raise ValueError("lower_bound và upper_bound phải là scalar hoặc vector đúng dimension.")
        if np.any(lb >= ub):
            raise ValueError("Mọi lower_bound phải nhỏ hơn upper_bound.")
        return lb, ub

    def _repair(self, candidate: Array) -> Array:
        assert self._lb is not None and self._ub is not None
        return np.clip(candidate, self._lb, self._ub)


# ---------------------------------------------------------------------------
# Demo chạy độc lập. Khi đưa vào framework, có thể bỏ phần này.
# ---------------------------------------------------------------------------
def sphere(x: Array) -> float:
    return float(np.sum(x**2))


if __name__ == "__main__":
    optimizer = RAPOParallelHybrid(
        population_size=50,
        max_iterations=100,
        lower_bound=-100.0,
        upper_bound=100.0,
        poa_mode="both_best",
        max_evaluations=None,
        seed=42,
    )
    result = optimizer.optimize(objective_func=sphere, dimension=30)

    print("Algorithm:", result.algorithm_name)
    print("Best fitness:", result.best_fitness)
    print("Best solution head:", result.best_solution[:5])
    print("NFE:", result.nfe)
    print("Runtime seconds:", round(result.runtime_seconds, 6))
    print("Stop reason:", result.metadata["stop_reason"])
    print("Winner count:", result.metadata["winner_count"])
