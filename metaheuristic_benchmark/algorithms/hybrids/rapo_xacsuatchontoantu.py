"""
RAPO Probabilistic / Roulette Operator Selection (ARO-POA)
================================================================

Biến thể lai xác suất chọn toán tử cho bài toán minimization liên tục.
Ở mỗi vòng lặp, thuật toán dùng xác suất p_ARO(t) để chọn khối cập nhật ARO,
ngược lại dùng POA:

    r ~ U(0, 1)
    nếu r < p_ARO(t)  -> dùng ARO
    ngược lại         -> dùng POA

Khác với AOS (Adaptive Operator Selection), xác suất ở file này KHÔNG cập nhật
bằng reward/cải thiện fitness. Xác suất có thể cố định hoặc biến thiên tuyến tính
theo thời gian thông qua p_aro_start và p_aro_end.

File viết dạng standalone để dễ kiểm thử. Khi tích hợp vào framework thật, có thể:
    - thay self._evaluate(...) bằng evaluator.evaluate(...)
    - thay OptimizationResult bằng class result chung của project
    - cho class kế thừa BaseOptimizer nếu framework bắt buộc.

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
    """Kết quả tối thiểu cần cho benchmark framework."""

    algorithm_name: str
    best_solution: Array
    best_fitness: float
    convergence_curve: List[float]
    nfe: int
    runtime_seconds: float
    metadata: Dict[str, object] = field(default_factory=dict)


class RAPOProbabilisticRoulette:
    """
    RAPO lai xác suất chọn toán tử - Probabilistic / Roulette Operator Selection.

    Ý tưởng chính:
        1. Khởi tạo chung một quần thể X.
        2. Ở mỗi vòng lặp t, dùng roulette với xác suất p_ARO(t).
        3. Nếu chọn ARO:
              - dùng Energy Shrink A(t) để chọn ARO Detour hoặc ARO Random Hiding.
           Nếu chọn POA:
              - dùng phase POA theo poa_phase_mode.
        4. Repair biên, tính fitness, greedy selection.
        5. Lưu X_best và convergence_curve.

    Tham số chính:
        population_size     : số cá thể N.
        max_iterations      : số vòng lặp T.
        lower_bound         : cận dưới, scalar hoặc vector d chiều.
        upper_bound         : cận trên, scalar hoặc vector d chiều.
        p_aro_start         : xác suất chọn ARO ở vòng lặp đầu.
        p_aro_end           : xác suất chọn ARO ở vòng lặp cuối.
                              Nếu bằng p_aro_start -> xác suất cố định.
        selection_level     : "iteration" hoặc "individual".
                              - iteration: mỗi vòng lặp chọn ARO/POA một lần cho cả quần thể.
                              - individual: mỗi cá thể tự roulette chọn ARO/POA.
        poa_phase_mode      : "energy", "attack", "defense", hoặc "both".
                              - energy : A(t)>1 dùng Attack, ngược lại Defense.
                              - attack : chỉ dùng POA Phase 1.
                              - defense: chỉ dùng POA Phase 2.
                              - both   : chạy Phase 1 rồi Phase 2 như POA gốc.
        max_evaluations     : ngân sách đánh giá hàm mục tiêu để so sánh công bằng.
        seed                : random seed.
    """

    ALGORITHM_NAME = "RAPO_Probabilistic_Roulette"

    def __init__(
        self,
        population_size: int = 50,
        max_iterations: int = 100,
        lower_bound: float | Sequence[float] = -100.0,
        upper_bound: float | Sequence[float] = 100.0,
        p_aro_start: float = 0.5,
        p_aro_end: Optional[float] = None,
        selection_level: str = "iteration",
        poa_phase_mode: str = "energy",
        energy_threshold: float = 1.0,
        max_evaluations: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size phải >= 2.")
        if max_iterations < 1:
            raise ValueError("max_iterations phải >= 1.")
        if not 0.0 <= p_aro_start <= 1.0:
            raise ValueError("p_aro_start phải nằm trong [0, 1].")
        if p_aro_end is None:
            p_aro_end = p_aro_start
        if not 0.0 <= p_aro_end <= 1.0:
            raise ValueError("p_aro_end phải nằm trong [0, 1].")
        if selection_level not in {"iteration", "individual"}:
            raise ValueError('selection_level chỉ nhận "iteration" hoặc "individual".')
        if poa_phase_mode not in {"energy", "attack", "defense", "both"}:
            raise ValueError('poa_phase_mode chỉ nhận "energy", "attack", "defense", hoặc "both".')
        if max_evaluations is not None and max_evaluations < population_size:
            raise ValueError("max_evaluations phải >= population_size để đánh giá quần thể ban đầu.")

        self.population_size = int(population_size)
        self.max_iterations = int(max_iterations)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.p_aro_start = float(p_aro_start)
        self.p_aro_end = float(p_aro_end)
        self.selection_level = selection_level
        self.poa_phase_mode = poa_phase_mode
        self.energy_threshold = float(energy_threshold)
        self.max_evaluations = max_evaluations
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.nfe = 0
        self._lb: Array | None = None
        self._ub: Array | None = None

    def optimize(self, objective_func: ObjectiveFunction, dimension: int) -> OptimizationResult:
        """Chạy tối ưu hóa một hàm mục tiêu liên tục."""

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

        algorithm_usage = {"ARO": 0, "POA": 0}
        algorithm_success = {"ARO": 0, "POA": 0}
        operator_usage = {
            "ARO_DETOUR": 0,
            "ARO_HIDING": 0,
            "POA_ATTACK": 0,
            "POA_DEFENSE": 0,
        }
        operator_success = {key: 0 for key in operator_usage}
        iteration_trace: List[Dict[str, object]] = []

        stop_reason = "max_iterations"

        for t in range(1, self.max_iterations + 1):
            if self._budget_exhausted():
                stop_reason = "max_evaluations"
                break

            p_aro_t = self._p_aro_at(t)
            iteration_algorithm: Optional[str] = None
            if self.selection_level == "iteration":
                iteration_algorithm = self._roulette_algorithm(p_aro_t)

            iter_usage = {"ARO": 0, "POA": 0}
            iter_success = {"ARO": 0, "POA": 0}
            iter_operator_usage = {key: 0 for key in operator_usage}
            iter_operator_success = {key: 0 for key in operator_success}

            for i in range(self.population_size):
                if self._budget_exhausted():
                    stop_reason = "max_evaluations"
                    break

                chosen_algorithm = (
                    iteration_algorithm
                    if iteration_algorithm is not None
                    else self._roulette_algorithm(p_aro_t)
                )
                if chosen_algorithm is None:
                    raise RuntimeError("Không chọn được thuật toán trong roulette.")

                algorithm_usage[chosen_algorithm] += 1
                iter_usage[chosen_algorithm] += 1

                if chosen_algorithm == "ARO":
                    operator_name, candidate = self._aro_candidate(
                        population=population,
                        index=i,
                        iteration=t,
                        dimension=dimension,
                    )
                    operator_usage[operator_name] += 1
                    iter_operator_usage[operator_name] += 1
                    improved = self._try_update(
                        objective_func=objective_func,
                        population=population,
                        fitness=fitness,
                        index=i,
                        candidate=candidate,
                    )
                    if improved:
                        algorithm_success[chosen_algorithm] += 1
                        iter_success[chosen_algorithm] += 1
                        operator_success[operator_name] += 1
                        iter_operator_success[operator_name] += 1

                else:  # chosen_algorithm == "POA"
                    if self.poa_phase_mode == "both":
                        # Phase 1: Predator Attack
                        if not self._budget_exhausted():
                            operator_name = "POA_ATTACK"
                            candidate = self._poa_attack_candidate(
                                population=population,
                                fitness=fitness,
                                index=i,
                                iteration=t,
                            )
                            operator_usage[operator_name] += 1
                            iter_operator_usage[operator_name] += 1
                            improved = self._try_update(
                                objective_func=objective_func,
                                population=population,
                                fitness=fitness,
                                index=i,
                                candidate=candidate,
                            )
                            if improved:
                                algorithm_success[chosen_algorithm] += 1
                                iter_success[chosen_algorithm] += 1
                                operator_success[operator_name] += 1
                                iter_operator_success[operator_name] += 1

                        # Phase 2: Defense Mechanism
                        if not self._budget_exhausted():
                            operator_name = "POA_DEFENSE"
                            candidate = self._poa_defense_candidate(
                                population=population,
                                index=i,
                                iteration=t,
                            )
                            operator_usage[operator_name] += 1
                            iter_operator_usage[operator_name] += 1
                            improved = self._try_update(
                                objective_func=objective_func,
                                population=population,
                                fitness=fitness,
                                index=i,
                                candidate=candidate,
                            )
                            if improved:
                                algorithm_success[chosen_algorithm] += 1
                                iter_success[chosen_algorithm] += 1
                                operator_success[operator_name] += 1
                                iter_operator_success[operator_name] += 1
                    else:
                        operator_name, candidate = self._poa_candidate(
                            population=population,
                            fitness=fitness,
                            index=i,
                            iteration=t,
                        )
                        operator_usage[operator_name] += 1
                        iter_operator_usage[operator_name] += 1
                        improved = self._try_update(
                            objective_func=objective_func,
                            population=population,
                            fitness=fitness,
                            index=i,
                            candidate=candidate,
                        )
                        if improved:
                            algorithm_success[chosen_algorithm] += 1
                            iter_success[chosen_algorithm] += 1
                            operator_success[operator_name] += 1
                            iter_operator_success[operator_name] += 1

                best_idx = int(np.argmin(fitness))
                if float(fitness[best_idx]) <= best_fitness:
                    best_fitness = float(fitness[best_idx])
                    best_solution = population[best_idx].copy()

            convergence_curve.append(best_fitness)
            iteration_trace.append(
                {
                    "iteration": t,
                    "p_aro": p_aro_t,
                    "p_poa": 1.0 - p_aro_t,
                    "selection_level": self.selection_level,
                    "iteration_algorithm": iteration_algorithm,
                    "best_fitness": best_fitness,
                    "nfe": self.nfe,
                    "algorithm_usage": iter_usage,
                    "algorithm_success": iter_success,
                    "operator_usage": iter_operator_usage,
                    "operator_success": iter_operator_success,
                }
            )

            if stop_reason == "max_evaluations":
                break

        runtime_seconds = perf_counter() - start_time
        return OptimizationResult(
            algorithm_name=self.ALGORITHM_NAME,
            best_solution=best_solution,
            best_fitness=best_fitness,
            convergence_curve=convergence_curve,
            nfe=self.nfe,
            runtime_seconds=runtime_seconds,
            metadata={
                "hybrid_type": "Probabilistic / Roulette Operator Selection",
                "roulette_rule": "r < p_ARO(t) -> ARO, else POA",
                "p_aro_start": self.p_aro_start,
                "p_aro_end": self.p_aro_end,
                "selection_level": self.selection_level,
                "poa_phase_mode": self.poa_phase_mode,
                "energy_threshold": self.energy_threshold,
                "algorithm_usage": algorithm_usage,
                "algorithm_success": algorithm_success,
                "operator_usage": operator_usage,
                "operator_success": operator_success,
                "iteration_trace": iteration_trace,
                "stop_reason": stop_reason,
                "seed": self.seed,
            },
        )

    # ------------------------------------------------------------------
    # Roulette selection
    # ------------------------------------------------------------------

    def _p_aro_at(self, iteration: int) -> float:
        """Xác suất chọn ARO tại vòng lặp t."""

        if self.max_iterations <= 1:
            return float(np.clip(self.p_aro_end, 0.0, 1.0))
        progress = (iteration - 1) / (self.max_iterations - 1)
        p = self.p_aro_start + (self.p_aro_end - self.p_aro_start) * progress
        return float(np.clip(p, 0.0, 1.0))

    def _roulette_algorithm(self, p_aro: float) -> str:
        """Chọn ARO hoặc POA bằng roulette."""

        return "ARO" if self.rng.random() < p_aro else "POA"

    # ------------------------------------------------------------------
    # ARO operators
    # ------------------------------------------------------------------

    def _aro_candidate(
        self,
        population: Array,
        index: int,
        iteration: int,
        dimension: int,
    ) -> Tuple[str, Array]:
        """Sinh candidate bằng ARO Detour hoặc ARO Random Hiding theo Energy Shrink."""

        energy = self._aro_energy(iteration)
        if energy > self.energy_threshold:
            return "ARO_DETOUR", self._aro_detour_candidate(population, index, iteration, dimension)
        return "ARO_HIDING", self._aro_hiding_candidate(population, index, iteration, dimension)

    def _aro_energy(self, iteration: int) -> float:
        """Energy Shrink của ARO: A(t)=4(1-t/T)ln(1/r)."""

        r = max(float(self.rng.random()), 1e-12)
        return float(4.0 * (1.0 - iteration / self.max_iterations) * np.log(1.0 / r))

    def _aro_running_operator(self, iteration: int, dimension: int) -> Array:
        """Running operator R = L * c dùng trong ARO."""

        r2 = float(self.rng.random())
        r3 = float(self.rng.random())
        progress = (iteration - 1) / max(self.max_iterations, 1)
        running_length = (np.e - np.exp(progress**2)) * np.sin(2.0 * np.pi * r2)

        c = np.zeros(dimension, dtype=float)
        selected_count = int(np.ceil(r3 * dimension))
        selected_count = min(max(selected_count, 1), dimension)
        selected_dims = self.rng.choice(dimension, size=selected_count, replace=False)
        c[selected_dims] = 1.0
        return running_length * c

    def _aro_detour_candidate(
        self,
        population: Array,
        index: int,
        iteration: int,
        dimension: int,
    ) -> Array:
        """ARO Detour Foraging - exploration."""

        candidates = np.delete(np.arange(self.population_size), index)
        j = int(self.rng.choice(candidates))
        r1 = float(self.rng.random())
        r_operator = self._aro_running_operator(iteration, dimension)
        perturb_flag = round(0.5 * (0.05 + r1))
        perturbation = perturb_flag * self.rng.normal(0.0, 1.0, size=dimension)
        return population[j] + r_operator * (population[index] - population[j]) + perturbation

    def _aro_hiding_candidate(
        self,
        population: Array,
        index: int,
        iteration: int,
        dimension: int,
    ) -> Array:
        """ARO Random Hiding - exploitation."""

        hiding_parameter = ((self.max_iterations - iteration + 1.0) / self.max_iterations) * self.rng.random()
        gr = np.zeros(dimension, dtype=float)
        selected_dim = int(self.rng.integers(0, dimension))
        gr[selected_dim] = 1.0
        burrow = population[index] + hiding_parameter * gr * population[index]
        r_operator = self._aro_running_operator(iteration, dimension)
        r4 = float(self.rng.random())
        return population[index] + r_operator * (r4 * burrow - population[index])

    # ------------------------------------------------------------------
    # POA operators
    # ------------------------------------------------------------------

    def _poa_candidate(
        self,
        population: Array,
        fitness: Array,
        index: int,
        iteration: int,
    ) -> Tuple[str, Array]:
        """Sinh candidate POA theo cấu hình poa_phase_mode."""

        if self.poa_phase_mode == "attack":
            return "POA_ATTACK", self._poa_attack_candidate(population, fitness, index, iteration)
        if self.poa_phase_mode == "defense":
            return "POA_DEFENSE", self._poa_defense_candidate(population, index, iteration)

        # poa_phase_mode == "energy"
        energy = self._aro_energy(iteration)
        if energy > self.energy_threshold:
            return "POA_ATTACK", self._poa_attack_candidate(population, fitness, index, iteration)
        return "POA_DEFENSE", self._poa_defense_candidate(population, index, iteration)

    def _poa_attack_candidate(
        self,
        population: Array,
        fitness: Array,
        index: int,
        iteration: int,
    ) -> Array:
        """POA Phase 1 - Predator Attack towards Pufferfish."""

        better_indices = np.where(fitness < fitness[index])[0]
        better_indices = better_indices[better_indices != index]

        if better_indices.size == 0:
            # Cá thể tốt nhất không có CP_i. Dùng Defense làm fallback để tránh đứng yên.
            return self._poa_defense_candidate(population, index, iteration)

        selected_index = int(self.rng.choice(better_indices))
        selected_pufferfish = population[selected_index]
        r = self.rng.random(size=population.shape[1])
        integer_factor = self.rng.integers(1, 3, size=population.shape[1])  # 1 hoặc 2
        return population[index] + r * (selected_pufferfish - integer_factor * population[index])

    def _poa_defense_candidate(
        self,
        population: Array,
        index: int,
        iteration: int,
    ) -> Array:
        """POA Phase 2 - Defense Mechanism."""

        if self._lb is None or self._ub is None:
            raise RuntimeError("Bounds chưa được khởi tạo.")
        r = self.rng.random(size=population.shape[1])
        step = (1.0 - 2.0 * r) * (self._ub - self._lb) / max(iteration, 1)
        return population[index] + step

    # ------------------------------------------------------------------
    # Common utilities
    # ------------------------------------------------------------------

    def _try_update(
        self,
        objective_func: ObjectiveFunction,
        population: Array,
        fitness: Array,
        index: int,
        candidate: Array,
    ) -> bool:
        """Repair, evaluate, greedy selection. Trả về True nếu cải thiện."""

        candidate = self._repair(candidate)
        candidate_fitness = self._evaluate(objective_func, candidate)
        if candidate_fitness <= fitness[index]:
            population[index] = candidate
            fitness[index] = candidate_fitness
            return True
        return False

    def _normalize_bounds(self, dimension: int) -> Tuple[Array, Array]:
        lb = np.asarray(self.lower_bound, dtype=float)
        ub = np.asarray(self.upper_bound, dtype=float)

        if lb.ndim == 0:
            lb = np.full(dimension, float(lb))
        if ub.ndim == 0:
            ub = np.full(dimension, float(ub))

        if lb.shape != (dimension,) or ub.shape != (dimension,):
            raise ValueError("lower_bound và upper_bound phải là scalar hoặc vector có độ dài bằng dimension.")
        if np.any(lb >= ub):
            raise ValueError("Mọi phần tử lower_bound phải nhỏ hơn upper_bound.")
        return lb, ub

    def _repair(self, x: Array) -> Array:
        if self._lb is None or self._ub is None:
            raise RuntimeError("Bounds chưa được khởi tạo.")
        return np.clip(x, self._lb, self._ub)

    def _evaluate(self, objective_func: ObjectiveFunction, x: Array) -> float:
        self.nfe += 1
        value = float(objective_func(x))
        if not np.isfinite(value):
            return float("inf")
        return value

    def _budget_exhausted(self) -> bool:
        return self.max_evaluations is not None and self.nfe >= self.max_evaluations


# ----------------------------------------------------------------------
# Demo chạy độc lập
# ----------------------------------------------------------------------


def sphere(x: Array) -> float:
    """Benchmark Sphere: global minimum = 0 tại x = 0."""

    return float(np.sum(x**2))


if __name__ == "__main__":
    optimizer = RAPOProbabilisticRoulette(
        population_size=50,
        max_iterations=100,
        lower_bound=-100.0,
        upper_bound=100.0,
        p_aro_start=0.6,
        p_aro_end=0.4,
        selection_level="iteration",
        poa_phase_mode="energy",
        seed=42,
    )
    result = optimizer.optimize(objective_func=sphere, dimension=30)

    print("Algorithm:", result.algorithm_name)
    print("Best fitness:", result.best_fitness)
    print("Best solution first 5 dims:", result.best_solution[:5])
    print("NFE:", result.nfe)
    print("Runtime seconds:", round(result.runtime_seconds, 6))
    print("Algorithm usage:", result.metadata["algorithm_usage"])
    print("Operator usage:", result.metadata["operator_usage"])
    print("Stop reason:", result.metadata["stop_reason"])
