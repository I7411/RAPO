"""
rapo_sequential.py

RAPO Sequential Hybrid (High-level Relay / Sequential Hybrid)
================================================================

Cơ chế lai:
    - 50% số vòng lặp đầu chạy thuật toán thứ nhất.
    - 50% số vòng lặp sau chạy thuật toán còn lại.
    - Mặc định: ARO -> POA, đúng với sơ đồ RAPO tuần tự ARO/POA.
    - Có thể đảo chiều bằng order="POA_ARO".

Bài toán:
    - Tối ưu hóa liên tục.
    - Minimization.
    - Có đếm số lần đánh giá hàm mục tiêu: nfe.
    - Có tùy chọn max_evaluations để dừng theo evaluation budget.

Gợi ý đặt file trong framework:
    metaheuristic_benchmark/algorithms/hybrids/rapo_sequential.py

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


class RAPOSequential:
    """
    RAPO Sequential Hybrid: lai tuần tự theo thời gian giữa ARO và POA.

    Thứ tự mặc định:
        ARO chạy ở giai đoạn đầu, POA chạy ở giai đoạn sau.

    Nếu dùng order="POA_ARO":
        POA chạy ở giai đoạn đầu, ARO chạy ở giai đoạn sau.

    Ghi chú công bằng so sánh:
        ARO thường đánh giá 1 nghiệm ứng viên / cá thể / vòng lặp.
        POA có 2 pha nên có thể đánh giá tối đa 2 nghiệm ứng viên / cá thể / vòng lặp.
        Vì vậy nên so sánh bằng max_evaluations nếu muốn công bằng nghiêm ngặt theo NFE.
    """

    def __init__(
        self,
        population_size: int = 50,
        max_iterations: int = 100,
        lower_bound: float | Sequence[float] = -100.0,
        upper_bound: float | Sequence[float] = 100.0,
        switch_ratio: float = 0.5,
        order: str = "ARO_POA",
        seed: Optional[int] = None,
        max_evaluations: Optional[int] = None,
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size phải >= 2.")
        if max_iterations < 1:
            raise ValueError("max_iterations phải >= 1.")
        if not (0.0 < switch_ratio < 1.0):
            raise ValueError("switch_ratio phải nằm trong khoảng (0, 1).")

        order = order.upper().strip()
        if order not in {"ARO_POA", "POA_ARO"}:
            raise ValueError('order chỉ nhận "ARO_POA" hoặc "POA_ARO".')

        self.population_size = int(population_size)
        self.max_iterations = int(max_iterations)
        self.lower_bound_input = lower_bound
        self.upper_bound_input = upper_bound
        self.switch_ratio = float(switch_ratio)
        self.order = order
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
        Chạy RAPO Sequential trên một hàm mục tiêu.

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

        try:
            for i in range(self.population_size):
                fitness[i] = self._evaluate(objective_func, population[i])
        except EvaluationBudgetExceeded:
            evaluated_count = max(self.nfe, 1)
            fitness[self.nfe :] = np.inf
            population[self.nfe :] = population[0]

        best_index = int(np.argmin(fitness))
        best_solution = population[best_index].copy()
        best_fitness = float(fitness[best_index])

        convergence: List[float] = []
        iteration_trace: List[Dict[str, object]] = []
        switch_iteration = max(1, int(np.floor(self.max_iterations * self.switch_ratio)))

        stopped_by_budget = False

        for iteration in range(1, self.max_iterations + 1):
            phase_algorithm = self._select_phase_algorithm(iteration, switch_iteration)

            try:
                for i in range(self.population_size):
                    if phase_algorithm == "ARO":
                        candidate, candidate_fitness = self._aro_update_one(
                            objective_func=objective_func,
                            population=population,
                            fitness=fitness,
                            index=i,
                            iteration=iteration,
                            lower_bound=lb,
                            upper_bound=ub,
                        )
                    else:
                        candidate, candidate_fitness = self._poa_update_one(
                            objective_func=objective_func,
                            population=population,
                            fitness=fitness,
                            index=i,
                            iteration=iteration,
                            lower_bound=lb,
                            upper_bound=ub,
                        )

                    population[i] = candidate
                    fitness[i] = candidate_fitness

                    if candidate_fitness < best_fitness:
                        best_fitness = float(candidate_fitness)
                        best_solution = candidate.copy()

            except EvaluationBudgetExceeded:
                stopped_by_budget = True

            convergence.append(best_fitness)
            iteration_trace.append(
                {
                    "iteration": iteration,
                    "phase_algorithm": phase_algorithm,
                    "best_fitness": best_fitness,
                    "nfe": self.nfe,
                }
            )

            if stopped_by_budget:
                break

        runtime = perf_counter() - start_time

        return OptimizationResult(
            algorithm=f"RAPO_Sequential_{self.order}",
            best_solution=best_solution,
            best_fitness=best_fitness,
            convergence=convergence,
            nfe=self.nfe,
            runtime_seconds=runtime,
            metadata={
                "population_size": self.population_size,
                "max_iterations": self.max_iterations,
                "dimension": dimension,
                "switch_ratio": self.switch_ratio,
                "switch_iteration": switch_iteration,
                "order": self.order,
                "seed": self.seed,
                "max_evaluations": self.max_evaluations,
                "stopped_by_budget": stopped_by_budget,
                "minimization": True,
                "phase_rule": self._phase_rule_text(switch_iteration),
                "iteration_trace": iteration_trace,
            },
        )

    def _select_phase_algorithm(self, iteration: int, switch_iteration: int) -> str:
        first, second = self.order.split("_")
        return first if iteration <= switch_iteration else second

    def _phase_rule_text(self, switch_iteration: int) -> str:
        first, second = self.order.split("_")
        return (
            f"{first}, 1 <= t <= {switch_iteration}; "
            f"{second}, {switch_iteration + 1} <= t <= {self.max_iterations}"
        )

    def _aro_update_one(
        self,
        objective_func: ObjectiveFunction,
        population: np.ndarray,
        fitness: np.ndarray,
        index: int,
        iteration: int,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Một bước cập nhật ARO cho cá thể index.

        ARO dùng Energy Shrink để chọn:
            - Detour Foraging nếu A(t) > 1.
            - Random Hiding nếu A(t) <= 1.

        Sau khi sinh nghiệm ứng viên, dùng greedy selection:
            nếu f(candidate) <= f(current) thì nhận candidate, ngược lại giữ current.
        """
        current = population[index]
        current_fitness = float(fitness[index])
        dimension = current.size

        energy = self._aro_energy(iteration)

        if energy > 1.0:
            candidate = self._aro_detour_foraging(population, index, iteration)
        else:
            candidate = self._aro_random_hiding(population, index, iteration)

        candidate = self._repair_bound(candidate, lower_bound, upper_bound)
        candidate_fitness = self._evaluate(objective_func, candidate)

        if candidate_fitness <= current_fitness:
            return candidate, candidate_fitness
        return current.copy(), current_fitness

    def _aro_energy(self, iteration: int) -> float:
        # r thuộc (0, 1), tránh log(1/0).
        r = float(self.rng.uniform(1e-12, 1.0))
        return 4.0 * (1.0 - iteration / self.max_iterations) * np.log(1.0 / r)

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

        return other + running_operator * (current - other) + perturb_switch * gaussian_noise

    def _aro_random_hiding(
        self,
        population: np.ndarray,
        index: int,
        iteration: int,
    ) -> np.ndarray:
        _, dimension = population.shape
        current = population[index]

        selected_dimension = int(self.rng.integers(0, dimension))
        mask = np.zeros(dimension, dtype=float)
        mask[selected_dimension] = 1.0

        r4 = float(self.rng.random())
        hiding_parameter = ((self.max_iterations - iteration + 1.0) / self.max_iterations) * r4
        selected_burrow = current + hiding_parameter * mask * current

        running_operator = self._aro_running_operator(dimension, iteration)
        return current + running_operator * (r4 * selected_burrow - current)

    def _aro_running_operator(self, dimension: int, iteration: int) -> np.ndarray:
        r2 = float(self.rng.random())
        r3 = float(self.rng.random())

        # Running length L trong ARO: giảm dần theo thời gian, có dao động sin.
        normalized_time = ((iteration - 1.0) / self.max_iterations) ** 2
        running_length = (np.e - np.exp(normalized_time)) * np.sin(2.0 * np.pi * r2)

        selected_count = max(1, int(np.ceil(r3 * dimension)))
        selected_dimensions = self.rng.choice(dimension, size=selected_count, replace=False)

        mapping_vector = np.zeros(dimension, dtype=float)
        mapping_vector[selected_dimensions] = 1.0

        return running_length * mapping_vector

    def _poa_update_one(
        self,
        objective_func: ObjectiveFunction,
        population: np.ndarray,
        fitness: np.ndarray,
        index: int,
        iteration: int,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Một bước cập nhật POA-Pufferfish cho cá thể index.

        Pha 1 - Predator Attack / Exploration:
            CP_i = tập các cá thể tốt hơn X_i.
            Nếu CP_i không rỗng, chọn SP_i ngẫu nhiên từ CP_i và sinh Y.
            Nếu Y tốt hơn thì nhận Y.

        Pha 2 - Defense Mechanism / Exploitation:
            Sinh Z bằng bước nhỏ phụ thuộc (ub - lb) / t.
            Nếu Z tốt hơn thì nhận Z.
        """
        current = population[index].copy()
        current_fitness = float(fitness[index])
        dimension = current.size

        better_indices = np.flatnonzero(fitness < current_fitness)
        better_indices = better_indices[better_indices != index]

        if better_indices.size > 0:
            selected_index = int(self.rng.choice(better_indices))
            selected_pufferfish = population[selected_index]

            random_vector = self.rng.random(size=dimension)
            integer_vector = self.rng.integers(1, 3, size=dimension)  # 1 hoặc 2

            y = current + random_vector * (selected_pufferfish - integer_vector * current)
            y = self._repair_bound(y, lower_bound, upper_bound)
            y_fitness = self._evaluate(objective_func, y)

            if y_fitness <= current_fitness:
                current = y
                current_fitness = y_fitness

        random_vector = self.rng.random(size=dimension)
        z = current + (1.0 - 2.0 * random_vector) * ((upper_bound - lower_bound) / iteration)
        z = self._repair_bound(z, lower_bound, upper_bound)
        z_fitness = self._evaluate(objective_func, z)

        if z_fitness <= current_fitness:
            return z, z_fitness
        return current.copy(), current_fitness

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
    optimizer = RAPOSequential(
        population_size=30,
        max_iterations=100,
        lower_bound=-100.0,
        upper_bound=100.0,
        switch_ratio=0.5,
        order="ARO_POA",  # đổi thành "POA_ARO" nếu muốn đảo thứ tự
        seed=42,
        max_evaluations=None,  # đặt ví dụ 3000 nếu muốn dừng theo NFE budget
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
