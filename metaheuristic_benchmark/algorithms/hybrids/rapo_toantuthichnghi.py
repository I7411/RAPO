"""
rapo_operator_selection.py

RAPO Adaptive Operator Selection (AOS)
======================================

Cơ chế lai:
    - Lai chọn toán tử thích nghi giữa ARO và POA-Pufferfish.
    - Tại mỗi bước cập nhật cá thể, thuật toán xác định xác suất chọn toán tử:
          p_ARO(t), p_POA(t)
      dựa trên điểm tín dụng/hiệu quả cải thiện fitness của từng toán tử.
    - Sinh r ~ U(0, 1):
          nếu r < p_ARO(t) -> dùng toán tử ARO để sinh nghiệm ứng viên V_i;
          ngược lại       -> dùng toán tử POA để sinh nghiệm ứng viên V_i.
    - Sau đó repair biên, tính f(V_i), greedy selection, cập nhật X_best.
    - Credit assignment: toán tử nào tạo nghiệm cải thiện fitness sẽ được tăng reward,
      từ đó xác suất chọn toán tử đó tăng ở các bước sau.

Bài toán:
    - Tối ưu hóa liên tục.
    - Minimization.
    - Có đếm số lần đánh giá hàm mục tiêu: nfe.
    - Có tùy chọn max_evaluations để dừng theo evaluation budget.
    - Có seed để tái lập kết quả.

Gợi ý đặt file trong framework:
    metaheuristic_benchmark/algorithms/hybrids/rapo_operator_selection.py

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


class RAPOOperatorSelection:
    """
    RAPO AOS: Adaptive Operator Selection giữa ARO và POA-Pufferfish.

    Ý tưởng:
        Không chia cứng theo thời gian, cũng không dùng một ngưỡng energy cố định
        để chọn thuật toán. Thay vào đó, xác suất chọn ARO/POA được cập nhật theo
        hiệu quả thực tế của từng toán tử trong quá trình tìm kiếm.

    Xác suất chọn toán tử:
        q_ARO(t), q_POA(t): điểm tín dụng thích nghi của từng toán tử.

        p_ARO(t) = p_min + (1 - 2*p_min) * q_ARO(t) / (q_ARO(t) + q_POA(t))
        p_POA(t) = 1 - p_ARO(t)

        p_min giúp mọi toán tử luôn còn cơ hội được chọn, tránh bị loại sớm.

    Credit assignment:
        improvement = max(0, f_old - f_new)
        relative_improvement = improvement / (abs(f_old) + eps)
        reward = min(relative_improvement, reward_clip) + beta * success

        q_selected = (1 - alpha) * q_selected + alpha * reward

    Trong nhánh ARO:
        A(t) > 1  -> Detour Foraging
        A(t) <= 1 -> Random Hiding

    Trong nhánh POA:
        phase_rule='energy':
            A(t) > 1  -> Predator Attack nếu CP_i không rỗng, ngược lại Defense
            A(t) <= 1 -> Defense Mechanism
        phase_rule='time':
            nửa đầu vòng lặp -> Predator Attack nếu CP_i không rỗng, ngược lại Defense
            nửa sau          -> Defense Mechanism

    Ghi chú công bằng:
        Mỗi lần cập nhật cá thể chỉ sinh và đánh giá 1 nghiệm ứng viên V_i.
        Do đó NFE xấp xỉ population_size + population_size * max_iterations,
        trừ khi dừng sớm bằng max_evaluations.
    """

    def __init__(
        self,
        population_size: int = 50,
        max_iterations: int = 100,
        lower_bound: float | Sequence[float] = -100.0,
        upper_bound: float | Sequence[float] = 100.0,
        p_aro_initial: float = 0.5,
        p_poa_initial: float = 0.5,
        alpha: float = 0.2,
        beta: float = 0.05,
        min_probability: float = 0.05,
        reward_clip: float = 1.0,
        phase_rule: str = "energy",
        seed: Optional[int] = None,
        max_evaluations: Optional[int] = None,
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size phải >= 2.")
        if max_iterations < 1:
            raise ValueError("max_iterations phải >= 1.")
        if p_aro_initial < 0 or p_poa_initial < 0 or (p_aro_initial + p_poa_initial) <= 0:
            raise ValueError("p_aro_initial và p_poa_initial phải không âm và có tổng > 0.")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha phải nằm trong khoảng (0, 1].")
        if beta < 0:
            raise ValueError("beta phải >= 0.")
        if not (0.0 <= min_probability < 0.5):
            raise ValueError("min_probability phải nằm trong khoảng [0, 0.5).")
        if reward_clip <= 0:
            raise ValueError("reward_clip phải > 0.")
        if max_evaluations is not None and max_evaluations < population_size:
            raise ValueError("max_evaluations phải >= population_size để đánh giá quần thể ban đầu.")

        phase_rule = phase_rule.lower().strip()
        if phase_rule not in {"energy", "time"}:
            raise ValueError('phase_rule chỉ nhận "energy" hoặc "time".')

        prob_sum = float(p_aro_initial + p_poa_initial)
        self.p_aro_initial = float(p_aro_initial / prob_sum)
        self.p_poa_initial = float(p_poa_initial / prob_sum)

        self.population_size = int(population_size)
        self.max_iterations = int(max_iterations)
        self.lower_bound_input = lower_bound
        self.upper_bound_input = upper_bound
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.min_probability = float(min_probability)
        self.reward_clip = float(reward_clip)
        self.phase_rule = phase_rule
        self.seed = seed
        self.max_evaluations = max_evaluations

        self.rng = np.random.default_rng(seed)
        self.nfe = 0

        self.credit = {
            "ARO": self.p_aro_initial,
            "POA": self.p_poa_initial,
        }

    def optimize(
        self,
        objective_func: ObjectiveFunction,
        dimension: int,
        lower_bound: float | Sequence[float] | None = None,
        upper_bound: float | Sequence[float] | None = None,
    ) -> OptimizationResult:
        """
        Chạy RAPO AOS trên một hàm mục tiêu.

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
        self.credit = {
            "ARO": self.p_aro_initial,
            "POA": self.p_poa_initial,
        }

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
        probability_trace: List[Dict[str, float]] = []
        stopped_by_budget = False

        for iteration in range(1, self.max_iterations + 1):
            usage_count = {"ARO": 0, "POA": 0}
            success_count = {"ARO": 0, "POA": 0}
            reward_sum = {"ARO": 0.0, "POA": 0.0}
            improvement_sum = {"ARO": 0.0, "POA": 0.0}

            for i in range(self.population_size):
                p_aro, p_poa = self._operator_probabilities()
                random_selector = float(self.rng.random())

                current = population[i].copy()
                current_fitness = float(fitness[i])

                if random_selector < p_aro:
                    selected_operator = "ARO"
                    candidate = self._aro_candidate(
                        population=population,
                        index=i,
                        iteration=iteration,
                    )
                else:
                    selected_operator = "POA"
                    candidate = self._poa_candidate(
                        population=population,
                        fitness=fitness,
                        index=i,
                        iteration=iteration,
                        lower_bound=lb,
                        upper_bound=ub,
                    )

                usage_count[selected_operator] += 1
                candidate = self._repair_bound(candidate, lb, ub)

                try:
                    candidate_fitness = self._evaluate(objective_func, candidate)
                except EvaluationBudgetExceeded:
                    stopped_by_budget = True
                    break

                improvement = max(0.0, current_fitness - candidate_fitness)
                accepted = candidate_fitness <= current_fitness

                if accepted:
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    if improvement > 0.0:
                        success_count[selected_operator] += 1
                    if candidate_fitness < best_fitness:
                        best_fitness = float(candidate_fitness)
                        best_solution = candidate.copy()
                else:
                    population[i] = current
                    fitness[i] = current_fitness

                reward = self._compute_reward(
                    old_fitness=current_fitness,
                    new_fitness=candidate_fitness,
                    accepted=accepted,
                )
                reward_sum[selected_operator] += reward
                improvement_sum[selected_operator] += improvement
                self._update_credit(selected_operator, reward)

                p_aro_after, p_poa_after = self._operator_probabilities()
                probability_trace.append(
                    {
                        "iteration": float(iteration),
                        "individual_index": float(i),
                        "p_aro": float(p_aro_after),
                        "p_poa": float(p_poa_after),
                        "credit_aro": float(self.credit["ARO"]),
                        "credit_poa": float(self.credit["POA"]),
                    }
                )

            convergence.append(best_fitness)
            p_aro_end, p_poa_end = self._operator_probabilities()
            iteration_trace.append(
                {
                    "iteration": iteration,
                    "p_aro": float(p_aro_end),
                    "p_poa": float(p_poa_end),
                    "credit_aro": float(self.credit["ARO"]),
                    "credit_poa": float(self.credit["POA"]),
                    "aro_usage": usage_count["ARO"],
                    "poa_usage": usage_count["POA"],
                    "aro_success": success_count["ARO"],
                    "poa_success": success_count["POA"],
                    "aro_reward_sum": float(reward_sum["ARO"]),
                    "poa_reward_sum": float(reward_sum["POA"]),
                    "aro_improvement_sum": float(improvement_sum["ARO"]),
                    "poa_improvement_sum": float(improvement_sum["POA"]),
                    "best_fitness": best_fitness,
                    "nfe": self.nfe,
                }
            )

            if stopped_by_budget:
                break

        runtime = perf_counter() - start_time

        return OptimizationResult(
            algorithm="RAPO_Operator_Selection_AOS_ARO_POA",
            best_solution=best_solution,
            best_fitness=best_fitness,
            convergence=convergence,
            nfe=self.nfe,
            runtime_seconds=runtime,
            metadata={
                "population_size": self.population_size,
                "max_iterations": self.max_iterations,
                "dimension": dimension,
                "p_aro_initial": self.p_aro_initial,
                "p_poa_initial": self.p_poa_initial,
                "alpha": self.alpha,
                "beta": self.beta,
                "min_probability": self.min_probability,
                "reward_clip": self.reward_clip,
                "phase_rule": self.phase_rule,
                "seed": self.seed,
                "max_evaluations": self.max_evaluations,
                "stopped_by_budget": stopped_by_budget,
                "minimization": True,
                "selection_rule": "Chọn ARO nếu r < p_ARO(t), ngược lại chọn POA.",
                "probability_formula": "p_ARO = p_min + (1 - 2*p_min) * q_ARO / (q_ARO + q_POA)",
                "credit_formula": "q_selected = (1 - alpha) * q_selected + alpha * reward",
                "reward_formula": "reward = min(relative_improvement, reward_clip) + beta * success",
                "iteration_trace": iteration_trace,
                "probability_trace": probability_trace,
            },
        )

    # ------------------------------------------------------------------
    # Adaptive Operator Selection
    # ------------------------------------------------------------------

    def _operator_probabilities(self) -> Tuple[float, float]:
        q_aro = max(float(self.credit["ARO"]), 0.0)
        q_poa = max(float(self.credit["POA"]), 0.0)
        total = q_aro + q_poa

        if total <= 1e-15:
            raw_p_aro = self.p_aro_initial
        else:
            raw_p_aro = q_aro / total

        p_aro = self.min_probability + (1.0 - 2.0 * self.min_probability) * raw_p_aro
        p_aro = float(np.clip(p_aro, self.min_probability, 1.0 - self.min_probability))
        p_poa = 1.0 - p_aro
        return p_aro, p_poa

    def _compute_reward(self, old_fitness: float, new_fitness: float, accepted: bool) -> float:
        if not np.isfinite(old_fitness) and np.isfinite(new_fitness):
            relative_improvement = self.reward_clip
        elif not np.isfinite(old_fitness) or not np.isfinite(new_fitness):
            relative_improvement = 0.0
        else:
            improvement = max(0.0, old_fitness - new_fitness)
            denominator = abs(old_fitness) + 1e-12
            relative_improvement = improvement / denominator

        relative_improvement = float(np.clip(relative_improvement, 0.0, self.reward_clip))
        success = 1.0 if accepted and new_fitness < old_fitness else 0.0
        return relative_improvement + self.beta * success

    def _update_credit(self, selected_operator: str, reward: float) -> None:
        old_credit = float(self.credit[selected_operator])
        new_credit = (1.0 - self.alpha) * old_credit + self.alpha * float(reward)
        self.credit[selected_operator] = max(new_credit, 0.0)

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
        Sinh V_i bằng toán tử ARO.

        A(t) > 1  -> Detour Foraging.
        A(t) <= 1 -> Random Hiding.
        """
        energy = self._aro_energy(iteration)
        if energy > 1.0:
            return self._aro_detour_foraging(population, index, iteration)
        return self._aro_random_hiding(population, index, iteration)

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

    def _aro_random_hiding(
        self,
        population: np.ndarray,
        index: int,
        iteration: int,
    ) -> np.ndarray:
        current = population[index]
        dimension = current.size

        hiding_parameter = ((self.max_iterations - iteration + 1.0) / self.max_iterations) * float(self.rng.random())

        selected_dimension = int(self.rng.integers(0, dimension))
        burrow_mask = np.zeros(dimension, dtype=float)
        burrow_mask[selected_dimension] = 1.0

        selected_burrow = current + hiding_parameter * burrow_mask * current
        running_operator = self._aro_running_operator(dimension, iteration)
        r4 = float(self.rng.random())

        candidate = current + running_operator * (r4 * selected_burrow - current)
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
        Sinh V_i bằng toán tử POA-Pufferfish.

        Phase 1 - Predator Attack:
            CP_i = {X_k | F_k < F_i, k != i}
            Chọn SP_i ngẫu nhiên từ CP_i và sinh candidate.

        Phase 2 - Defense Mechanism:
            Sinh candidate bằng bước nhỏ quanh vị trí hiện tại.
        """
        current = population[index]
        current_fitness = float(fitness[index])

        use_attack = False
        if self.phase_rule == "energy":
            use_attack = self._aro_energy(iteration) > 1.0
        elif self.phase_rule == "time":
            use_attack = iteration <= int(np.ceil(0.5 * self.max_iterations))

        if use_attack:
            better_indices = np.flatnonzero(fitness < current_fitness)
            better_indices = better_indices[better_indices != index]

            if better_indices.size > 0:
                selected_index = int(self.rng.choice(better_indices))
                selected_pufferfish = population[selected_index]
                return self._poa_predator_attack(current, selected_pufferfish)

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
    optimizer = RAPOOperatorSelection(
        population_size=30,
        max_iterations=100,
        lower_bound=-100.0,
        upper_bound=100.0,
        p_aro_initial=0.5,
        p_poa_initial=0.5,
        alpha=0.2,
        beta=0.05,
        min_probability=0.05,
        phase_rule="energy",  # energy | time
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
    print("Final p_ARO:", result.metadata["iteration_trace"][-1]["p_aro"])
    print("Final p_POA:", result.metadata["iteration_trace"][-1]["p_poa"])
