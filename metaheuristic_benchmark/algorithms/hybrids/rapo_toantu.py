"""
RAPO Operator-level Hybridization (ARO-POA)
================================================

Biến thể lai theo toán tử cho bài toán minimization liên tục.
Mỗi cá thể chọn một toán tử trong nhóm:
    1. ARO-Detour        : toán tử khám phá của ARO
    2. ARO-Hiding        : toán tử khai thác của ARO
    3. POA-Attack        : Phase 1 - Predator Attack của Pufferfish POA
    4. POA-Defense       : Phase 2 - Defense Mechanism của Pufferfish POA

File này được viết dạng standalone để dễ kiểm thử. Khi tích hợp vào framework thật,
có thể thay hàm self._evaluate(...) bằng evaluator.evaluate(...), và thay
OptimizationResult bằng class result chung của project.

Yêu cầu: Python 3.11+, numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


Array = np.ndarray
ObjectiveFunction = Callable[[Array], float]


@dataclass
class OptimizationResult:
    """Kết quả tối ưu hóa tối thiểu cần cho benchmark framework."""

    algorithm_name: str
    best_solution: Array
    best_fitness: float
    convergence_curve: List[float]
    nfe: int
    runtime_seconds: float
    metadata: Dict[str, object] = field(default_factory=dict)


class RAPOOperatorLevelHybrid:
    """
    RAPO theo toán tử - Operator-level Hybridization.

    Ý tưởng:
        Với mỗi cá thể Xi ở mỗi vòng lặp t, chọn một toán tử theo xác suất P:
            - ARO_DETOUR
            - ARO_HIDING
            - POA_ATTACK
            - POA_DEFENSE
        Sinh nghiệm ứng viên Vi, sửa biên, đánh giá fitness, rồi greedy selection.

    Tham số chính:
        population_size      : số cá thể N
        max_iterations       : số vòng lặp T
        lower_bound          : cận dưới, scalar hoặc vector d chiều
        upper_bound          : cận trên, scalar hoặc vector d chiều
        operator_probabilities:
            Xác suất chọn 4 toán tử theo thứ tự:
            [ARO_DETOUR, ARO_HIDING, POA_ATTACK, POA_DEFENSE]
            Nếu None, dùng xác suất đều 0.25 cho mỗi toán tử.
        max_evaluations      : ngân sách đánh giá hàm mục tiêu, dùng để so sánh công bằng
        seed                 : random seed để tái lập kết quả
    """

    OPERATOR_NAMES: Tuple[str, str, str, str] = (
        "ARO_DETOUR",
        "ARO_HIDING",
        "POA_ATTACK",
        "POA_DEFENSE",
    )

    def __init__(
        self,
        population_size: int = 50,
        max_iterations: int = 100,
        lower_bound: float | Sequence[float] = -100.0,
        upper_bound: float | Sequence[float] = 100.0,
        operator_probabilities: Optional[Sequence[float]] = None,
        max_evaluations: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size phải >= 2.")
        if max_iterations < 1:
            raise ValueError("max_iterations phải >= 1.")

        self.population_size = int(population_size)
        self.max_iterations = int(max_iterations)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_evaluations = max_evaluations
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if operator_probabilities is None:
            self.operator_probabilities = np.full(4, 0.25, dtype=float)
        else:
            probs = np.asarray(operator_probabilities, dtype=float)
            if probs.shape != (4,):
                raise ValueError(
                    "operator_probabilities phải có đúng 4 giá trị: "
                    "[ARO_DETOUR, ARO_HIDING, POA_ATTACK, POA_DEFENSE]."
                )
            if np.any(probs < 0):
                raise ValueError("operator_probabilities không được chứa giá trị âm.")
            total = float(np.sum(probs))
            if total <= 0:
                raise ValueError("Tổng operator_probabilities phải > 0.")
            self.operator_probabilities = probs / total

        self.nfe = 0
        self._lb: Array | None = None
        self._ub: Array | None = None

    def optimize(self, objective_func: ObjectiveFunction, dimension: int) -> OptimizationResult:
        """Chạy tối ưu hóa cho một objective function liên tục."""

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

        operator_usage = {name: 0 for name in self.OPERATOR_NAMES}
        operator_success = {name: 0 for name in self.OPERATOR_NAMES}
        operator_improvement = {name: 0.0 for name in self.OPERATOR_NAMES}
        iteration_trace: List[Dict[str, object]] = []

        stop_reason = "max_iterations"

        for t in range(1, self.max_iterations + 1):
            if self._budget_exhausted():
                stop_reason = "max_evaluations"
                break

            iter_usage = {name: 0 for name in self.OPERATOR_NAMES}
            iter_success = {name: 0 for name in self.OPERATOR_NAMES}

            for i in range(self.population_size):
                if self._budget_exhausted():
                    stop_reason = "max_evaluations"
                    break

                operator_name = self._select_operator()
                operator_usage[operator_name] += 1
                iter_usage[operator_name] += 1

                candidate = self._apply_operator(
                    operator_name=operator_name,
                    population=population,
                    fitness=fitness,
                    index=i,
                    iteration=t,
                    dimension=dimension,
                    best_solution=best_solution,
                )
                candidate = self._repair(candidate)
                candidate_fitness = self._evaluate(objective_func, candidate)

                old_fitness = float(fitness[i])
                if candidate_fitness <= old_fitness:
                    improvement = max(0.0, old_fitness - candidate_fitness)
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    operator_success[operator_name] += 1
                    iter_success[operator_name] += 1
                    operator_improvement[operator_name] += improvement

                    if candidate_fitness <= best_fitness:
                        best_fitness = float(candidate_fitness)
                        best_solution = candidate.copy()

            convergence_curve.append(best_fitness)
            iteration_trace.append(
                {
                    "iteration": t,
                    "best_fitness": best_fitness,
                    "nfe": self.nfe,
                    "operator_usage": iter_usage,
                    "operator_success": iter_success,
                }
            )

            if stop_reason == "max_evaluations":
                break

        runtime_seconds = perf_counter() - start_time
        return OptimizationResult(
            algorithm_name="RAPO_Operator_Level_Hybrid",
            best_solution=best_solution,
            best_fitness=best_fitness,
            convergence_curve=convergence_curve,
            nfe=self.nfe,
            runtime_seconds=runtime_seconds,
            metadata={
                "hybrid_type": "Operator-level Hybridization",
                "operator_set": list(self.OPERATOR_NAMES),
                "operator_probabilities": {
                    name: float(prob)
                    for name, prob in zip(self.OPERATOR_NAMES, self.operator_probabilities)
                },
                "operator_usage": operator_usage,
                "operator_success": operator_success,
                "operator_improvement": operator_improvement,
                "population_size": self.population_size,
                "max_iterations": self.max_iterations,
                "max_evaluations": self.max_evaluations,
                "dimension": dimension,
                "seed": self.seed,
                "stop_reason": stop_reason,
                "iteration_trace": iteration_trace,
            },
        )

    def _select_operator(self) -> str:
        idx = int(self.rng.choice(4, p=self.operator_probabilities))
        return self.OPERATOR_NAMES[idx]

    def _apply_operator(
        self,
        operator_name: str,
        population: Array,
        fitness: Array,
        index: int,
        iteration: int,
        dimension: int,
        best_solution: Array,
    ) -> Array:
        if operator_name == "ARO_DETOUR":
            return self._aro_detour_foraging(population, index, iteration, dimension)
        if operator_name == "ARO_HIDING":
            return self._aro_random_hiding(population, index, iteration, dimension)
        if operator_name == "POA_ATTACK":
            return self._poa_predator_attack(population, fitness, index, best_solution)
        if operator_name == "POA_DEFENSE":
            return self._poa_defense_mechanism(population[index], iteration)
        raise ValueError(f"Toán tử không hợp lệ: {operator_name}")

    def _aro_detour_foraging(
        self,
        population: Array,
        index: int,
        iteration: int,
        dimension: int,
    ) -> Array:
        """ARO Detour Foraging - toán tử exploration."""

        n = population.shape[0]
        xi = population[index]

        j = int(self.rng.integers(0, n - 1))
        if j >= index:
            j += 1
        xj = population[j]

        r1 = float(self.rng.random())
        r2 = float(self.rng.random())
        r3 = float(self.rng.random())

        # Running length. Dạng này giữ tinh thần ARO: bước lớn đầu kỳ, giảm dần về cuối.
        time_ratio = (iteration - 1) / max(1, self.max_iterations)
        length = (np.e - np.exp(time_ratio**2)) * np.sin(2.0 * np.pi * r2)

        # Mapping vector c: chọn ngẫu nhiên một tập chiều để nhiễu.
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
        """ARO Random Hiding - toán tử exploitation."""

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
        length = (np.e - np.exp(((iteration - 1) / max(1, self.max_iterations)) ** 2)) * np.sin(
            2.0 * np.pi * r5
        )

        c = np.zeros(dimension, dtype=float)
        selected_count = max(1, int(np.ceil(float(self.rng.random()) * dimension)))
        selected_dims = self.rng.choice(dimension, size=selected_count, replace=False)
        c[selected_dims] = 1.0
        running_operator = length * c

        return xi + running_operator * (r4 * random_burrow - xi)

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
            # Cá thể tốt nhất không có CP_i. Dùng X_best làm fallback để tránh trả về Xi nguyên trạng.
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

    def _evaluate(self, objective_func: ObjectiveFunction, solution: Array) -> float:
        if self._budget_exhausted():
            # Không nên gọi khi hết budget; nhánh này chỉ để bảo vệ.
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
    optimizer = RAPOOperatorLevelHybrid(
        population_size=50,
        max_iterations=100,
        lower_bound=-100.0,
        upper_bound=100.0,
        operator_probabilities=[0.25, 0.25, 0.25, 0.25],
        max_evaluations=None,
        seed=42,
    )
    result = optimizer.optimize(objective_func=sphere, dimension=30)

    print("Algorithm:", result.algorithm_name)
    print("Best fitness:", result.best_fitness)
    print("Best solution first 5 dims:", result.best_solution[:5])
    print("NFE:", result.nfe)
    print("Runtime seconds:", round(result.runtime_seconds, 6))
    print("Operator usage:", result.metadata["operator_usage"])
    print("Operator success:", result.metadata["operator_success"])
