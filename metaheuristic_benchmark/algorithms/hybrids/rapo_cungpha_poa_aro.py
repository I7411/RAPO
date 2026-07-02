"""
rapo_reverse_sequential_operator_chain_poa_aro.py

RAPO Reverse Sequential Operator Chain - POA -> ARO trong cùng pha
=================================================================

Cơ chế lai:
    - Đây là biến thể Reverse Sequential Operator Chain của RAPO.
    - Trong mỗi vòng lặp và với mỗi cá thể Xi, thuật toán dùng Energy Shrink
      của ARO để quyết định pha tìm kiếm:

        + Nếu A(t) > 1:
            POA Phase 1 - Predator Attack tạo nghiệm trung gian U_i
            ARO Detour Foraging tiếp tục từ U_i để tạo nghiệm cuối V_i

        + Nếu A(t) <= 1:
            POA Phase 2 - Defense Mechanism tạo nghiệm trung gian U_i
            ARO Random Hiding tiếp tục từ U_i để tạo nghiệm cuối V_i

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


class RAPOReverseSequentialOperatorChainPOAARO:
    """
    RAPO Reverse Sequential Operator Chain theo hướng POA -> ARO.

    Mỗi cá thể được cập nhật bằng một chuỗi toán tử trong cùng pha:
        - Pha khám phá: POA Predator Attack -> ARO Detour Foraging
        - Pha khai thác: POA Defense Mechanism -> ARO Random Hiding

    Parameters
    ----------
    population_size:
        Số cá thể N.
    max_iterations:
        Số vòng lặp tối đa T.
    lower_bound, upper_bound:
        Cận dưới và cận trên. Có thể là scalar hoặc vector độ dài dimension.
    repair_intermediate:
        Nếu True, nghiệm trung gian U_i được repair trước khi đưa sang toán tử ARO.
        Nên để True để tránh ARO nhận một nghiệm trung gian vượt biên quá xa.
    poa_attack_fallback:
        Cách xử lý khi CP_i rỗng trong POA Attack:
        - "best": dùng X_best hiện tại làm SP_i;
        - "random": chọn ngẫu nhiên một cá thể khác;
        - "keep": giữ nguyên base Xi làm SP_i, làm bước POA Attack yếu hơn.
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
        poa_attack_fallback: str = "best",
        max_evaluations: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size phải >= 2.")
        if max_iterations < 1:
            raise ValueError("max_iterations phải >= 1.")
        if poa_attack_fallback not in {"best", "random", "keep"}:
            raise ValueError("poa_attack_fallback phải thuộc {'best', 'random', 'keep'}.")
        if max_evaluations is not None and max_evaluations < population_size:
            raise ValueError("max_evaluations nên >= population_size để đủ đánh giá ban đầu.")

        self.population_size = int(population_size)
        self.max_iterations = int(max_iterations)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.repair_intermediate = bool(repair_intermediate)
        self.poa_attack_fallback = poa_attack_fallback
        self.max_evaluations = max_evaluations
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.nfe = 0
        self._lb: Array | None = None
        self._ub: Array | None = None

    def optimize(self, objective_func: ObjectiveFunc, dimension: int) -> OptimizationResult:
        """Chạy RAPO-RSOC POA->ARO trên một hàm mục tiêu liên tục."""
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
            "POA_ATTACK_TO_ARO_DETOUR": 0,
            "POA_DEFENSE_TO_ARO_HIDING": 0,
        }
        chain_success = {
            "POA_ATTACK_TO_ARO_DETOUR": 0,
            "POA_DEFENSE_TO_ARO_HIDING": 0,
        }

        stop_reason = "max_iterations"

        for t in range(1, self.max_iterations + 1):
            if self._budget_exhausted():
                stop_reason = "max_evaluations"
                break

            iteration_best_before = best_fitness
            iteration_success = {key: 0 for key in chain_success}
            iteration_usage = {key: 0 for key in chain_usage}

            # Tính một energy chung cho vòng lặp, đúng với sơ đồ RAPO-RSOC.
            energy = self._energy_factor(t)

            for i in range(self.population_size):
                if self._budget_exhausted():
                    stop_reason = "max_evaluations"
                    break

                current_fitness = float(fitness[i])

                if energy > 1.0:
                    chain_name = "POA_ATTACK_TO_ARO_DETOUR"

                    # Bước 1: POA Predator Attack tạo nghiệm trung gian U_i từ X_i.
                    intermediate = self._poa_predator_attack(
                        population=population,
                        fitness=fitness,
                        index=i,
                        best_solution=best_solution,
                    )
                    if self.repair_intermediate:
                        intermediate = self._repair(intermediate)

                    # Bước 2: ARO Detour Foraging tiếp tục từ U_i để tạo V_i.
                    candidate = self._aro_detour_from_base(
                        base=intermediate,
                        population=population,
                        index=i,
                        iteration=t,
                        dimension=dimension,
                    )
                else:
                    chain_name = "POA_DEFENSE_TO_ARO_HIDING"

                    # Bước 1: POA Defense Mechanism tạo nghiệm trung gian U_i từ X_i.
                    intermediate = self._poa_defense_mechanism(
                        base=population[i],
                        iteration=t,
                    )
                    if self.repair_intermediate:
                        intermediate = self._repair(intermediate)

                    # Bước 2: ARO Random Hiding tiếp tục từ U_i để tạo V_i.
                    candidate = self._aro_hiding_from_base(
                        base=intermediate,
                        iteration=t,
                        dimension=dimension,
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
            algorithm_name="RAPO_RSOC_POA_ARO",
            best_solution=best_solution,
            best_fitness=best_fitness,
            convergence_curve=convergence_curve,
            nfe=self.nfe,
            runtime_seconds=runtime_seconds,
            metadata={
                "hybrid_type": "Reverse Sequential Operator Chain",
                "chain_direction": "POA -> ARO",
                "exploration_chain": "POA Predator Attack -> ARO Detour Foraging",
                "exploitation_chain": "POA Defense Mechanism -> ARO Random Hiding",
                "population_size": self.population_size,
                "max_iterations": self.max_iterations,
                "repair_intermediate": self.repair_intermediate,
                "poa_attack_fallback": self.poa_attack_fallback,
                "max_evaluations": self.max_evaluations,
                "seed": self.seed,
                "chain_usage": chain_usage,
                "chain_success": chain_success,
                "iteration_trace": iteration_trace,
                "stop_reason": stop_reason,
            },
        )

    # ------------------------------------------------------------------
    # Toán tử POA-Pufferfish chạy trước trong chuỗi
    # ------------------------------------------------------------------
    def _poa_predator_attack(
        self,
        population: Array,
        fitness: Array,
        index: int,
        best_solution: Array,
    ) -> Array:
        """
        POA Phase 1 - Predator Attack tạo U_i từ X_i.

        U_i = X_i + r * (SP_i - I * X_i)
        với SP_i là cá thể tốt hơn X_i. Nếu CP_i rỗng, dùng fallback.
        """
        xi = population[index]
        better_indices = np.where(fitness < fitness[index])[0]
        better_indices = better_indices[better_indices != index]

        if better_indices.size > 0:
            selected_index = int(self.rng.choice(better_indices))
            selected_pufferfish = population[selected_index]
        elif self.poa_attack_fallback == "best":
            selected_pufferfish = best_solution
        elif self.poa_attack_fallback == "random":
            n = population.shape[0]
            j = int(self.rng.integers(0, n - 1))
            if j >= index:
                j += 1
            selected_pufferfish = population[j]
        else:  # keep
            selected_pufferfish = xi

        r = self.rng.random(size=xi.shape[0])
        integer_factor = self.rng.integers(1, 3, size=xi.shape[0])  # I thuộc {1, 2}
        return xi + r * (selected_pufferfish - integer_factor * xi)

    def _poa_defense_mechanism(self, base: Array, iteration: int) -> Array:
        """
        POA Phase 2 - Defense Mechanism tạo U_i từ X_i.

        U_i = X_i + (1 - 2r) * (ub - lb) / t
        """
        assert self._lb is not None and self._ub is not None
        r = self.rng.random(size=base.shape[0])
        step = (1.0 - 2.0 * r) * ((self._ub - self._lb) / max(1, iteration))
        return base + step

    # ------------------------------------------------------------------
    # Toán tử ARO chạy sau POA trong chuỗi
    # ------------------------------------------------------------------
    def _aro_detour_from_base(
        self,
        base: Array,
        population: Array,
        index: int,
        iteration: int,
        dimension: int,
    ) -> Array:
        """
        ARO Detour Foraging dùng base = U_i thay vì X_i.

        V_i = X_j + R * (U_i - X_j) + perturbation
        """
        n = population.shape[0]
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

        return xj + running_operator * (base - xj) + perturb_flag * gaussian_noise

    def _aro_hiding_from_base(self, base: Array, iteration: int, dimension: int) -> Array:
        """
        ARO Random Hiding dùng base = U_i thay vì X_i.

        V_i = U_i + R * (r4 * burrow(U_i) - U_i)
        """
        random_dim = int(self.rng.integers(0, dimension))

        gr = np.zeros(dimension, dtype=float)
        gr[random_dim] = 1.0

        hiding_factor = ((self.max_iterations - iteration + 1) / self.max_iterations) * float(
            self.rng.random()
        )
        random_burrow = base + hiding_factor * gr * base

        r4 = float(self.rng.random())
        r5 = float(self.rng.random())
        time_ratio = (iteration - 1) / max(1, self.max_iterations)
        length = (np.e - np.exp(time_ratio**2)) * np.sin(2.0 * np.pi * r5)

        c = np.zeros(dimension, dtype=float)
        selected_count = max(1, int(np.ceil(float(self.rng.random()) * dimension)))
        selected_dims = self.rng.choice(dimension, size=selected_count, replace=False)
        c[selected_dims] = 1.0

        running_operator = length * c
        return base + running_operator * (r4 * random_burrow - base)

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
    optimizer = RAPOReverseSequentialOperatorChainPOAARO(
        population_size=50,
        max_iterations=100,
        lower_bound=-100.0,
        upper_bound=100.0,
        repair_intermediate=True,
        poa_attack_fallback="best",
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
