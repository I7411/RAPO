"""
rapo_energy_switch.py

RAPO Energy Switch Hybrid (Adaptive Phase Switching)
=====================================================

Cơ chế lai:
    - Dùng hệ số năng lượng Energy Shrink của ARO để quyết định thuật toán cập nhật.
    - Nếu A(t) > energy_threshold: dùng toán tử ARO để sinh nghiệm ứng viên V_i.
    - Nếu A(t) <= energy_threshold: dùng toán tử POA-Pufferfish để sinh nghiệm ứng viên V_i.
    - Sau khi sinh V_i: repair biên, tính fitness, greedy selection.
    - Quần thể X được duy trì xuyên suốt, không khởi tạo lại khi chuyển pha.

Bài toán:
    - Tối ưu hóa liên tục.
    - Minimization.
    - Có đếm số lần đánh giá hàm mục tiêu: nfe.
    - Có tùy chọn max_evaluations để dừng theo evaluation budget.

Gợi ý đặt file trong framework:
    metaheuristic_benchmark/algorithms/hybrids/rapo_energy_switch.py

Yêu cầu:
    Python 3.11+
    numpy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


ObjectiveFunction = Callable[[np.ndarray], float]


class EvaluationBudgetExceeded(RuntimeError):
    """Raised internally when the maximum number of function evaluations is reached."""


@dataclass(slots=True)
class OptimizationResult:
    """Kết quả trả về sau khi tối ưu."""

    algorithm: str
    best_solution: np.ndarray
    best_fitness: float
    convergence: List[float]
    nfe: int
    runtime_seconds: float
    metadata: Dict[str, object] = field(default_factory=dict)


class RAPOEnergySwitch:
    """
    RAPO Adaptive Phase Switching dựa trên Energy Shrink của ARO.

    Quy tắc chuyển pha:
        A(t) = 4 * (1 - t / T) * ln(1 / r), với r thuộc (0, 1)

        Nếu A(t) > energy_threshold:
            Dùng ARO để cập nhật cá thể.

        Nếu A(t) <= energy_threshold:
            Dùng POA-Pufferfish để cập nhật cá thể.

    Ghi chú:
        File này triển khai đúng kiểu flowchart "mỗi nhánh sinh một nghiệm ứng viên V_i",
        sau đó mới repair, evaluate và greedy selection một lần.

        Vì POA gốc có hai pha, tham số poa_candidate_strategy cho phép chọn cách sinh V_i:
            - "auto"    : dùng POA Phase 1 - Predator Attack nếu CP_i không rỗng,
                          nếu CP_i rỗng thì fallback sang Phase 2 - Defense.
            - "attack"  : ưu tiên POA Phase 1, fallback sang Defense nếu CP_i rỗng.
            - "defense" : luôn dùng POA Phase 2 - Defense.

        Với cơ chế Energy Switch, lựa chọn "auto" là cân bằng nhất để vẫn giữ bản sắc POA
        nhưng không làm mỗi cá thể bị đánh giá 2 lần như POA đầy đủ.
    """

    def __init__(
        self,
        population_size: int = 50,
        max_iterations: int = 100,
        lower_bound: float | Sequence[float] = -100.0,
        upper_bound: float | Sequence[float] = 100.0,
        energy_threshold: float = 1.0,
        poa_candidate_strategy: str = "auto",
        seed: Optional[int] = None,
        max_evaluations: Optional[int] = None,
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size phải >= 2.")
        if max_iterations < 1:
            raise ValueError("max_iterations phải >= 1.")
        if energy_threshold < 0:
            raise ValueError("energy_threshold phải >= 0.")
        if max_evaluations is not None and max_evaluations < population_size:
            raise ValueError("max_evaluations phải >= population_size để đánh giá quần thể ban đầu.")

        poa_candidate_strategy = poa_candidate_strategy.lower().strip()
        if poa_candidate_strategy not in {"auto", "attack", "defense"}:
            raise ValueError('poa_candidate_strategy chỉ nhận "auto", "attack" hoặc "defense".')

        self.population_size = int(population_size)
        self.max_iterations = int(max_iterations)
        self.lower_bound_input = lower_bound
        self.upper_bound_input = upper_bound
        self.energy_threshold = float(energy_threshold)
        self.poa_candidate_strategy = poa_candidate_strategy
        self.seed = seed
        self.max_evaluations = max_evaluations

        self.rng = np.random.default_rng(seed)
        self.nfe = 0

    def optimize(
        self,
        objective_func: ObjectiveFunction,
        dimension: int,
        lower_bound: float | Sequence[float] | None = None,
        upper_bound: float | Sequence[float] | None = None,
    ) -> OptimizationResult:
        """
        Chạy RAPO Energy Switch trên một hàm mục tiêu.

        Parameters
        ----------
        objective_func:
            Hàm mục tiêu nhận vector numpy shape=(dimension,) và trả về fitness float.
        dimension:
            Số chiều bài toán.
        lower_bound, upper_bound:
            Biên dưới/biên trên. Nếu None thì dùng biên trong constructor.

        Returns
        -------
        OptimizationResult
            best_solution, best_fitness, convergence, nfe, runtime_seconds, metadata.
        """
        if dimension < 1:
            raise ValueError("dimension phải >= 1.")

        start_time = perf_counter()
        self.nfe = 0

        lb, ub = self._prepare_bounds(
            dimension=dimension,
            lower_bound=self.lower_bound_input if lower_bound is None else lower_bound,
            upper_bound=self.upper_bound_input if upper_bound is None else upper_bound,
        )

        population = self.rng.uniform(lb, ub, size=(self.population_size, dimension))
        fitness = np.empty(self.population_size, dtype=float)

        for i in range(self.population_size):
            fitness[i] = self._evaluate(objective_func, population[i])

        best_index = int(np.argmin(fitness))
        best_solution = population[best_index].copy()
        best_fitness = float(fitness[best_index])

        convergence: List[float] = []
        iteration_trace: List[Dict[str, object]] = []
        stopped_by_budget = False

        for iteration in range(1, self.max_iterations + 1):
            aro_count = 0
            poa_count = 0
            energy_values: List[float] = []

            for i in range(self.population_size):
                energy = self._aro_energy(iteration)
                energy_values.append(float(energy))

                current = population[i].copy()
                current_fitness = float(fitness[i])

                if energy > self.energy_threshold:
                    aro_count += 1
                    candidate = self._aro_candidate(
                        population=population,
                        index=i,
                        iteration=iteration,
                    )
                    selected_algorithm = "ARO"
                else:
                    poa_count += 1
                    candidate = self._poa_candidate(
                        population=population,
                        fitness=fitness,
                        index=i,
                        iteration=iteration,
                        lower_bound=lb,
                        upper_bound=ub,
                    )
                    selected_algorithm = "POA"

                candidate = self._repair_bound(candidate, lb, ub)

                try:
                    candidate_fitness = self._evaluate(objective_func, candidate)
                except EvaluationBudgetExceeded:
                    stopped_by_budget = True
                    break

                if candidate_fitness <= current_fitness:
                    population[i] = candidate
                    fitness[i] = candidate_fitness

                    if candidate_fitness < best_fitness:
                        best_fitness = float(candidate_fitness)
                        best_solution = candidate.copy()
                else:
                    population[i] = current
                    fitness[i] = current_fitness

                # selected_algorithm để dễ debug khi đặt breakpoint.
                _ = selected_algorithm

            convergence.append(best_fitness)
            iteration_trace.append(
                {
                    "iteration": iteration,
                    "aro_count": aro_count,
                    "poa_count": poa_count,
                    "energy_min": float(np.min(energy_values)) if energy_values else None,
                    "energy_mean": float(np.mean(energy_values)) if energy_values else None,
                    "energy_max": float(np.max(energy_values)) if energy_values else None,
                    "best_fitness": best_fitness,
                    "nfe": self.nfe,
                }
            )

            if stopped_by_budget:
                break

        runtime = perf_counter() - start_time

        return OptimizationResult(
            algorithm="RAPO_Energy_Switch_ARO_POA",
            best_solution=best_solution,
            best_fitness=best_fitness,
            convergence=convergence,
            nfe=self.nfe,
            runtime_seconds=runtime,
            metadata={
                "population_size": self.population_size,
                "max_iterations": self.max_iterations,
                "dimension": dimension,
                "energy_threshold": self.energy_threshold,
                "poa_candidate_strategy": self.poa_candidate_strategy,
                "seed": self.seed,
                "max_evaluations": self.max_evaluations,
                "stopped_by_budget": stopped_by_budget,
                "minimization": True,
                "phase_rule": "ARO nếu A(t) > energy_threshold, ngược lại POA",
                "energy_formula": "A(t) = 4 * (1 - t / T) * ln(1 / r)",
                "iteration_trace": iteration_trace,
            },
        )

    # ------------------------------------------------------------------
    # ARO branch
    # ------------------------------------------------------------------

    def _aro_energy(self, iteration: int) -> float:
        """
        Energy Shrink của ARO.

        A(t) = 4 * (1 - t / T) * ln(1 / r)
        r được lấy trong (0, 1), tránh r = 0 để không lỗi log.
        """
        r = float(self.rng.uniform(1e-12, 1.0))
        return float(4.0 * (1.0 - iteration / self.max_iterations) * np.log(1.0 / r))

    def _aro_candidate(
        self,
        population: np.ndarray,
        index: int,
        iteration: int,
    ) -> np.ndarray:
        """
        Sinh V_i bằng nhánh ARO.

        Vì Energy Switch đã quyết định A(t) > threshold mới dùng ARO,
        nhánh ARO ở đây dùng Detour Foraging để duy trì vai trò khám phá.
        """
        return self._aro_detour_foraging(population, index, iteration)

    def _aro_detour_foraging(
        self,
        population: np.ndarray,
        index: int,
        iteration: int,
    ) -> np.ndarray:
        n, dimension = population.shape

        j = int(self.rng.integers(0, n - 1))
        if j >= index:
            j += 1

        current = population[index]
        other = population[j]

        running_operator = self._aro_running_operator(dimension, iteration)
        r1 = float(self.rng.random())
        perturb_switch = round(0.5 * (0.05 + r1))
        gaussian_noise = self.rng.normal(0.0, 1.0, size=dimension)

        candidate = other + running_operator * (current - other) + perturb_switch * gaussian_noise
        return candidate

    def _aro_running_operator(self, dimension: int, iteration: int) -> np.ndarray:
        r2 = float(self.rng.random())
        r3 = float(self.rng.random())

        normalized_time = ((iteration - 1.0) / self.max_iterations) ** 2
        running_length = (np.e - np.exp(normalized_time)) * np.sin(2.0 * np.pi * r2)

        selected_count = max(1, int(np.ceil(r3 * dimension)))
        selected_dimensions = self.rng.choice(dimension, size=selected_count, replace=False)

        mapping_vector = np.zeros(dimension, dtype=float)
        mapping_vector[selected_dimensions] = 1.0

        return running_length * mapping_vector

    # ------------------------------------------------------------------
    # POA-Pufferfish branch
    # ------------------------------------------------------------------

    def _poa_candidate(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        index: int,
        iteration: int,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
    ) -> np.ndarray:
        """
        Sinh V_i bằng nhánh POA-Pufferfish.

        auto / attack:
            Phase 1 - Predator Attack:
                CP_i = {X_k | F_k < F_i, k != i}
                Chọn SP_i ngẫu nhiên từ CP_i và sinh candidate.

        defense:
            Phase 2 - Defense Mechanism:
                Sinh candidate bằng bước nhỏ quanh vị trí hiện tại.
        """
        current = population[index]
        current_fitness = float(fitness[index])

        if self.poa_candidate_strategy in {"auto", "attack"}:
            better_indices = np.flatnonzero(fitness < current_fitness)
            better_indices = better_indices[better_indices != index]

            if better_indices.size > 0:
                selected_index = int(self.rng.choice(better_indices))
                selected_pufferfish = population[selected_index]
                return self._poa_predator_attack(current, selected_pufferfish)

            # Nếu CP_i rỗng thì không có cá thể tốt hơn để làm SP_i.
            # Fallback sang defense để vẫn sinh được V_i hợp lệ.
            if self.poa_candidate_strategy == "attack":
                return self._poa_defense(current, iteration, lower_bound, upper_bound)

        return self._poa_defense(current, iteration, lower_bound, upper_bound)

    def _poa_predator_attack(
        self,
        current: np.ndarray,
        selected_pufferfish: np.ndarray,
    ) -> np.ndarray:
        dimension = current.size
        random_vector = self.rng.random(size=dimension)
        integer_vector = self.rng.integers(1, 3, size=dimension)  # 1 hoặc 2
        return current + random_vector * (selected_pufferfish - integer_vector * current)

    def _poa_defense(
        self,
        current: np.ndarray,
        iteration: int,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
    ) -> np.ndarray:
        dimension = current.size
        random_vector = self.rng.random(size=dimension)
        step = (upper_bound - lower_bound) / max(iteration, 1)
        return current + (1.0 - 2.0 * random_vector) * step

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _evaluate(self, objective_func: ObjectiveFunction, solution: np.ndarray) -> float:
        if self.max_evaluations is not None and self.nfe >= self.max_evaluations:
            raise EvaluationBudgetExceeded("Đã đạt max_evaluations.")

        value = float(objective_func(solution.copy()))
        self.nfe += 1

        if not np.isfinite(value):
            return float("inf")
        return value

    @staticmethod
    def _repair_bound(solution: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
        return np.clip(solution, lower_bound, upper_bound)

    @staticmethod
    def _prepare_bounds(
        dimension: int,
        lower_bound: float | Sequence[float],
        upper_bound: float | Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.asarray(lower_bound, dtype=float)
        ub = np.asarray(upper_bound, dtype=float)

        if lb.ndim == 0:
            lb = np.full(dimension, float(lb))
        if ub.ndim == 0:
            ub = np.full(dimension, float(ub))

        if lb.shape != (dimension,) or ub.shape != (dimension,):
            raise ValueError("lower_bound và upper_bound phải là số hoặc vector có độ dài bằng dimension.")
        if np.any(lb >= ub):
            raise ValueError("Mọi lower_bound phải nhỏ hơn upper_bound.")

        return lb, ub


# -----------------------------------------------------------------------------
# Demo chạy độc lập
# -----------------------------------------------------------------------------

def sphere(x: np.ndarray) -> float:
    """Sphere benchmark: global minimum f(0)=0."""
    return float(np.sum(x ** 2))


if __name__ == "__main__":
    optimizer = RAPOEnergySwitch(
        population_size=30,
        max_iterations=100,
        lower_bound=-100.0,
        upper_bound=100.0,
        energy_threshold=1.0,
        poa_candidate_strategy="auto",  # auto | attack | defense
        seed=42,
        max_evaluations=None,
    )

    result = optimizer.optimize(
        objective_func=sphere,
        dimension=30,
    )

    print("Algorithm:", result.algorithm)
    print("Best fitness:", result.best_fitness)
    print("Best solution:", result.best_solution)
    print("NFE:", result.nfe)
    print("Runtime seconds:", result.runtime_seconds)
    print("Phase rule:", result.metadata["phase_rule"])
    print("Energy formula:", result.metadata["energy_formula"])
