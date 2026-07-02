"""
rapo_sequential_operator_chain_aro_poa.py

RAPO Sequential Operator Chain - ARO -> POA trong cùng pha
==========================================================

Cơ chế lai:
    - Đây là biến thể Sequential Operator Chain của RAPO.
    - Trong mỗi vòng lặp và với mỗi cá thể Xi, thuật toán dùng Energy Shrink
      của ARO để quyết định pha tìm kiếm:

        + Nếu A(t) > 1:
            ARO Detour Foraging tạo nghiệm trung gian U_i
            POA Phase 1 - Predator Attack tiếp tục tinh chỉnh U_i để tạo V_i

        + Nếu A(t) <= 1:
            ARO Random Hiding tạo nghiệm trung gian U_i
            POA Phase 2 - Defense Mechanism tiếp tục tinh chỉnh U_i để tạo V_i

    - Chỉ nghiệm cuối V_i được đánh giá fitness và dùng greedy selection:
        nếu f(V_i) <= f(X_i) thì cập nhật X_i = V_i, ngược lại giữ nguyên X_i.
    - Bài toán mặc định: minimization.

Lưu ý tích hợp framework:
    Đây là bản standalone để dễ kiểm thử. Khi tích hợp vào framework thật,
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


class RAPOSequentialOperatorChainAROPOA:
    """
    RAPO Sequential Operator Chain theo hướng ARO -> POA.

    Mỗi cá thể được cập nhật bằng một chuỗi toán tử trong cùng pha:
        - Pha khám phá: ARO Detour -> POA Attack
        - Pha khai thác: ARO Hiding -> POA Defense

    Parameters
    ----------
    population_size:
        Số cá thể N.
    max_iterations:
        Số vòng lặp tối đa T.
    lower_bound, upper_bound:
        Cận dưới và cận trên. Có thể là scalar hoặc vector độ dài dimension.
    repair_intermediate:
        Nếu True, nghiệm trung gian U_i được repair trước khi đưa sang toán tử POA.
        Nên để True để tránh POA nhận một nghiệm trung gian vượt biên quá xa.
    max_evaluations:
        Ngân sách đánh giá hàm mục tiêu. Nên dùng khi so sánh công bằng theo NFE.
    seed:
        Random seed để tái lập kết quả.
    """

    def __init__(
        self,
        population_size: int = 50,
        max_iterations: int = 100,
        lower_bound: float | Sequence[float] | Array = -100.0,
        upper_bound: float | Sequence[float] | Array = 100.0,
        repair_intermediate: bool = True,
        max_evaluations: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size phải >= 2.")
        if max_iterations < 1:
            raise ValueError("max_iterations phải >= 1.")
        if max_evaluations is not None and max_evaluations < population_size:
            raise ValueError("max_evaluations nên >= population_size để đủ đánh giá ban đầu.")

        self.population_size = int(population_size)
        self.max_iterations = int(max_iterations)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.repair_intermediate = bool(repair_intermediate)
        self.max_evaluations = max_evaluations
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.nfe = 0
        self._lb: Array | None = None
        self._ub: Array | None = None

    def optimize(self, objective_func: ObjectiveFunc, dimension: int) -> OptimizationResult:
        """Chạy RAPO-SOC ARO->POA trên một hàm mục tiêu liên tục."""
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
        chain_usage = {
            "ARO_DETOUR_TO_POA_ATTACK": 0,
            "ARO_HIDING_TO_POA_DEFENSE": 0,
        }
        chain_success = {
            "ARO_DETOUR_TO_POA_ATTACK": 0,
            "ARO_HIDING_TO_POA_DEFENSE": 0,
        }

        stop_reason = "max_iterations"

        for t in range(1, self.max_iterations + 1):
            if self._budget_exhausted():
                stop_reason = "max_evaluations"
                break

            iteration_best_before = best_fitness
            iteration_success = {key: 0 for key in chain_success}
            iteration_usage = {key: 0 for key in chain_usage}

            # Tính một energy chung cho vòng lặp, đúng với sơ đồ RAPO-SOC.
            energy = self._energy_factor(t)

            for i in range(self.population_size):
                if self._budget_exhausted():
                    stop_reason = "max_evaluations"
                    break

                current_fitness = float(fitness[i])

                if energy > 1.0:
                    chain_name = "ARO_DETOUR_TO_POA_ATTACK"

                    # Bước 1: ARO Detour tạo nghiệm trung gian U_i từ X_i.
                    intermediate = self._aro_detour_foraging(
                        population=population,
                        index=i,
                        iteration=t,
                        dimension=dimension,
                    )
                    if self.repair_intermediate:
                        intermediate = self._repair(intermediate)

                    # Bước 2: POA Attack tiếp tục từ U_i để tạo nghiệm cuối V_i.
                    candidate = self._poa_predator_attack_from_base(
                        base=intermediate,
                        population=population,
                        fitness=fitness,
                        index=i,
                        best_solution=best_solution,
                    )
                else:
                    chain_name = "ARO_HIDING_TO_POA_DEFENSE"

                    # Bước 1: ARO Random Hiding tạo nghiệm trung gian U_i từ X_i.
                    intermediate = self._aro_random_hiding(
                        population=population,
                        index=i,
                        iteration=t,
                        dimension=dimension,
                    )
                    if self.repair_intermediate:
                        intermediate = self._repair(intermediate)

                    # Bước 2: POA Defense tiếp tục từ U_i để tạo nghiệm cuối V_i.
                    candidate = self._poa_defense_mechanism_from_base(
                        base=intermediate,
                        iteration=t,
                    )

                chain_usage[chain_name] += 1
                iteration_usage[chain_name] += 1

                candidate = self._repair(candidate)
                candidate_fitness = self._evaluate(objective_func, candidate)

                # Greedy selection cho minimization.
                if candidate_fitness <= current_fitness:
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    chain_success[chain_name] += 1
                    iteration_success[chain_name] += 1

                    if candidate_fitness <= best_fitness:
                        best_fitness = float(candidate_fitness)
                        best_solution = candidate.copy()

            convergence_curve.append(best_fitness)
            iteration_trace.append(
                {
                    "iteration": t,
                    "energy": float(energy),
                    "phase": "exploration_chain" if energy > 1.0 else "exploitation_chain",
                    "best_fitness_before": iteration_best_before,
                    "best_fitness_after": best_fitness,
                    "nfe": self.nfe,
                    "chain_usage": iteration_usage,
                    "chain_success": iteration_success,
                }
            )

            if stop_reason == "max_evaluations":
                break

        runtime_seconds = perf_counter() - start_time
        return OptimizationResult(
            algorithm_name="RAPO_SOC_ARO_POA",
            best_solution=best_solution,
            best_fitness=best_fitness,
            convergence_curve=convergence_curve,
            nfe=self.nfe,
            runtime_seconds=runtime_seconds,
            metadata={
                "hybrid_type": "Sequential Operator Chain",
                "chain_direction": "ARO -> POA",
                "exploration_chain": "ARO Detour Foraging -> POA Predator Attack",
                "exploitation_chain": "ARO Random Hiding -> POA Defense Mechanism",
                "population_size": self.population_size,
                "max_iterations": self.max_iterations,
                "repair_intermediate": self.repair_intermediate,
                "max_evaluations": self.max_evaluations,
                "seed": self.seed,
                "chain_usage": chain_usage,
                "chain_success": chain_success,
                "iteration_trace": iteration_trace,
                "stop_reason": stop_reason,
            },
        )

    # ------------------------------------------------------------------
    # Toán tử ARO
    # ------------------------------------------------------------------
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
    # Toán tử POA-Pufferfish chạy sau ARO trong chuỗi
    # ------------------------------------------------------------------
    def _poa_predator_attack_from_base(
        self,
        base: Array,
        population: Array,
        fitness: Array,
        index: int,
        best_solution: Array,
    ) -> Array:
        """
        POA Phase 1 - Predator Attack dùng base = U_i thay vì X_i.

        Công thức chain:
            V_i = U_i + r * (SP_i - I * U_i)
        trong đó SP_i chọn từ các cá thể tốt hơn X_i hiện tại.
        """
        better_indices = np.where(fitness < fitness[index])[0]
        better_indices = better_indices[better_indices != index]

        if better_indices.size > 0:
            selected_index = int(self.rng.choice(better_indices))
            selected_pufferfish = population[selected_index]
        else:
            selected_pufferfish = best_solution

        r = self.rng.random(size=base.shape[0])
        integer_factor = self.rng.integers(1, 3, size=base.shape[0])  # I thuộc {1, 2}
        return base + r * (selected_pufferfish - integer_factor * base)

    def _poa_defense_mechanism_from_base(self, base: Array, iteration: int) -> Array:
        """
        POA Phase 2 - Defense Mechanism dùng base = U_i thay vì X_i.

        Công thức chain:
            V_i = U_i + (1 - 2r) * (ub - lb) / t
        """
        assert self._lb is not None and self._ub is not None
        r = self.rng.random(size=base.shape[0])
        step = (1.0 - 2.0 * r) * ((self._ub - self._lb) / max(1, iteration))
        return base + step

    # ------------------------------------------------------------------
    # Hạ tầng dùng chung
    # ------------------------------------------------------------------
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
    optimizer = RAPOSequentialOperatorChainAROPOA(
        population_size=50,
        max_iterations=100,
        lower_bound=-100.0,
        upper_bound=100.0,
        repair_intermediate=True,
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
    print("Chain usage:", result.metadata["chain_usage"])
    print("Chain success:", result.metadata["chain_success"])
