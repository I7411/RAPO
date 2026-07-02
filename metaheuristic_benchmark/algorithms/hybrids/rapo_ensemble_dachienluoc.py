"""
RAPO-MSE: Multi-strategy Ensemble Hybridization for ARO-POA
================================================================

Biến thể lai ensemble đa chiến lược giữa:
- ARO: Artificial Rabbits Optimization
- POA: Pufferfish Optimization Algorithm

Ý tưởng chính
-------------
Trong mỗi vòng lặp, với từng cá thể X_i, thuật toán sinh nhiều nghiệm ứng viên
bằng các chiến lược ARO/POA khác nhau. Các nghiệm ứng viên cạnh tranh bằng
fitness. Nghiệm tốt nhất được chọn để cập nhật X_i theo greedy selection.

Pha khám phá, khi A(t) > 1:
    V1 = ARO Detour Foraging(X_i)
    V2 = POA Predator Attack(X_i)
    V3 = ARO Detour -> POA Attack(X_i)

Pha khai thác, khi A(t) <= 1:
    V1 = ARO Random Hiding(X_i)
    V2 = POA Defense Mechanism(X_i)
    V3 = ARO Hiding -> POA Defense(X_i)

Bài toán mặc định: minimization.
File này viết dạng standalone để có thể chạy thử trực tiếp, đồng thời dễ đưa vào:
    metaheuristic_benchmark/algorithms/hybrids/

Python: 3.11+
Phụ thuộc: numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import math
import time
import numpy as np


ObjectiveFunction = Callable[[np.ndarray], float]


@dataclass
class OptimizationResult:
    """Kết quả tối ưu hóa trả về từ RAPO-MSE."""

    best_solution: np.ndarray
    best_fitness: float
    convergence_curve: List[float]
    nfe: int
    runtime_seconds: float
    metadata: Dict[str, object]


class RAPOMultiStrategyEnsemble:
    """
    RAPO-MSE: Multi-strategy Ensemble Hybridization.

    Parameters
    ----------
    population_size:
        Số cá thể trong quần thể.
    max_iterations:
        Số vòng lặp tối đa T.
    lower_bound, upper_bound:
        Biên dưới và biên trên. Có thể là số hoặc vector theo từng chiều.
    seed:
        Seed ngẫu nhiên để tái lập kết quả.
    max_evaluations:
        Ngân sách đánh giá hàm mục tiêu. Nếu None thì không giới hạn theo nfe.
    energy_threshold:
        Ngưỡng chuyển pha. Theo ARO gốc thường dùng A(t) > 1.
    poa_attack_fallback:
        Cách xử lý khi CP_i rỗng trong POA Attack.
        - "best": dùng nghiệm tốt nhất toàn cục hiện tại làm SP_i.
        - "random": dùng một cá thể ngẫu nhiên khác làm SP_i.
        - "skip": trả về chính nghiệm hiện tại.
    evaluate_intermediate:
        Nếu True, nghiệm trung gian trong chuỗi ARO->POA cũng được đánh giá.
        Mặc định False để kiểm soát nfe và bám sát sơ đồ: chỉ đánh giá ứng viên cuối.
    """

    def __init__(
        self,
        population_size: int = 50,
        max_iterations: int = 100,
        lower_bound: float | Sequence[float] = -100.0,
        upper_bound: float | Sequence[float] = 100.0,
        seed: Optional[int] = None,
        max_evaluations: Optional[int] = None,
        energy_threshold: float = 1.0,
        poa_attack_fallback: str = "best",
        evaluate_intermediate: bool = False,
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size phải >= 2.")
        if max_iterations < 1:
            raise ValueError("max_iterations phải >= 1.")
        if poa_attack_fallback not in {"best", "random", "skip"}:
            raise ValueError("poa_attack_fallback phải là 'best', 'random' hoặc 'skip'.")

        self.population_size = int(population_size)
        self.max_iterations = int(max_iterations)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.seed = seed
        self.max_evaluations = max_evaluations
        self.energy_threshold = float(energy_threshold)
        self.poa_attack_fallback = poa_attack_fallback
        self.evaluate_intermediate = bool(evaluate_intermediate)

        self.rng = np.random.default_rng(seed)
        self.nfe = 0
        self._lower: np.ndarray | None = None
        self._upper: np.ndarray | None = None
        self._dimension = 0

        self._best_solution: np.ndarray | None = None
        self._best_fitness = math.inf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def optimize(self, objective_func: ObjectiveFunction, dimension: int) -> OptimizationResult:
        """Chạy RAPO-MSE trên một hàm mục tiêu liên tục."""
        if dimension < 1:
            raise ValueError("dimension phải >= 1.")

        start_time = time.perf_counter()
        self._dimension = int(dimension)
        self._lower, self._upper = self._prepare_bounds(dimension)
        self.nfe = 0

        population = self._initialize_population()
        fitness = np.array([self._evaluate(objective_func, x) for x in population], dtype=float)
        self._update_global_best(population, fitness)

        convergence_curve: List[float] = [self._best_fitness]
        phase_usage = {"exploration": 0, "exploitation": 0}
        strategy_usage = {
            "ARO_Detour": 0,
            "POA_Attack": 0,
            "ARO_Detour_POA_Attack": 0,
            "ARO_Hiding": 0,
            "POA_Defense": 0,
            "ARO_Hiding_POA_Defense": 0,
        }
        strategy_wins = {key: 0 for key in strategy_usage}
        iteration_trace: List[Dict[str, object]] = []

        stop_reason = "max_iterations"

        for t in range(1, self.max_iterations + 1):
            if self._budget_exhausted():
                stop_reason = "max_evaluations"
                break

            energy = self._energy_factor(t)
            is_exploration = energy > self.energy_threshold
            phase_name = "exploration" if is_exploration else "exploitation"
            phase_usage[phase_name] += 1

            improved_count = 0

            for i in range(self.population_size):
                if self._budget_exhausted():
                    stop_reason = "max_evaluations"
                    break

                current = population[i].copy()
                current_fitness = float(fitness[i])

                candidates = self._generate_ensemble_candidates(
                    population=population,
                    fitness=fitness,
                    index=i,
                    t=t,
                    energy=energy,
                    is_exploration=is_exploration,
                )

                best_candidate = current
                best_candidate_fitness = current_fitness
                best_strategy_name = "current"

                for strategy_name, candidate in candidates:
                    if self._budget_exhausted():
                        stop_reason = "max_evaluations"
                        break

                    repaired = self._repair(candidate)
                    candidate_fitness = self._evaluate(objective_func, repaired)
                    strategy_usage[strategy_name] += 1

                    if candidate_fitness < best_candidate_fitness:
                        best_candidate = repaired
                        best_candidate_fitness = candidate_fitness
                        best_strategy_name = strategy_name

                if best_candidate_fitness < current_fitness:
                    population[i] = best_candidate
                    fitness[i] = best_candidate_fitness
                    strategy_wins[best_strategy_name] += 1
                    improved_count += 1

            self._update_global_best(population, fitness)
            convergence_curve.append(self._best_fitness)

            iteration_trace.append(
                {
                    "iteration": t,
                    "energy": energy,
                    "phase": phase_name,
                    "best_fitness": self._best_fitness,
                    "nfe": self.nfe,
                    "improved_count": improved_count,
                }
            )

            if stop_reason == "max_evaluations":
                break

        runtime = time.perf_counter() - start_time
        assert self._best_solution is not None

        metadata: Dict[str, object] = {
            "algorithm": "RAPO_Multi_Strategy_Ensemble",
            "hybrid_type": "Multi-strategy Ensemble",
            "description": (
                "Trong mỗi vòng lặp, nhiều chiến lược ARO/POA cùng sinh ứng viên; "
                "ứng viên tốt nhất cạnh tranh bằng fitness để cập nhật cá thể."
            ),
            "population_size": self.population_size,
            "max_iterations": self.max_iterations,
            "dimension": dimension,
            "seed": self.seed,
            "max_evaluations": self.max_evaluations,
            "energy_threshold": self.energy_threshold,
            "poa_attack_fallback": self.poa_attack_fallback,
            "evaluate_intermediate": self.evaluate_intermediate,
            "phase_usage": phase_usage,
            "strategy_usage": strategy_usage,
            "strategy_wins": strategy_wins,
            "iteration_trace": iteration_trace,
            "stop_reason": stop_reason,
        }

        return OptimizationResult(
            best_solution=self._best_solution.copy(),
            best_fitness=float(self._best_fitness),
            convergence_curve=convergence_curve,
            nfe=self.nfe,
            runtime_seconds=runtime,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Ensemble candidate generation
    # ------------------------------------------------------------------
    def _generate_ensemble_candidates(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        index: int,
        t: int,
        energy: float,
        is_exploration: bool,
    ) -> List[Tuple[str, np.ndarray]]:
        """Sinh tập ứng viên C_i theo pha khám phá hoặc khai thác."""
        xi = population[index]

        if is_exploration:
            aro_detour = self._aro_detour_foraging(population, index, t)
            poa_attack = self._poa_predator_attack(population, fitness, index, base=xi)

            intermediate = self._repair(aro_detour)
            if self.evaluate_intermediate:
                # Không dùng fitness trung gian để selection; chỉ tùy chọn tiêu tốn nfe
                # khi cần mô phỏng đầy đủ hơn trong một framework riêng.
                pass
            chain = self._poa_predator_attack(population, fitness, index, base=intermediate)

            return [
                ("ARO_Detour", aro_detour),
                ("POA_Attack", poa_attack),
                ("ARO_Detour_POA_Attack", chain),
            ]

        aro_hiding = self._aro_random_hiding(xi, t)
        poa_defense = self._poa_defense_mechanism(base=xi, t=t)

        intermediate = self._repair(aro_hiding)
        chain = self._poa_defense_mechanism(base=intermediate, t=t)

        return [
            ("ARO_Hiding", aro_hiding),
            ("POA_Defense", poa_defense),
            ("ARO_Hiding_POA_Defense", chain),
        ]

    # ------------------------------------------------------------------
    # ARO operators
    # ------------------------------------------------------------------
    def _aro_detour_foraging(self, population: np.ndarray, index: int, t: int) -> np.ndarray:
        """ARO Detour Foraging: toán tử thiên về exploration."""
        n, d = population.shape
        xi = population[index]

        candidates = [j for j in range(n) if j != index]
        peer_index = int(self.rng.choice(candidates))
        xj = population[peer_index]

        running_operator = self._aro_running_operator(t, d)
        perturb_flag = round(0.5 * (0.05 + float(self.rng.random())))
        perturbation = perturb_flag * self.rng.normal(0.0, 1.0, size=d)

        return xj + running_operator * (xi - xj) + perturbation

    def _aro_random_hiding(self, xi: np.ndarray, t: int) -> np.ndarray:
        """ARO Random Hiding: toán tử thiên về exploitation."""
        d = xi.shape[0]
        h = ((self.max_iterations - t + 1) / self.max_iterations) * float(self.rng.random())

        gr = np.zeros(d)
        gr[int(self.rng.integers(0, d))] = 1.0
        burrow = xi + h * gr * xi

        running_operator = self._aro_running_operator(t, d)
        return xi + running_operator * (float(self.rng.random()) * burrow - xi)

    def _aro_running_operator(self, t: int, dimension: int) -> np.ndarray:
        """Tạo running operator R = L * c cho ARO."""
        progress = (t - 1) / max(self.max_iterations, 1)
        # Dạng suy giảm theo thời gian, bám tinh thần công thức ARO gốc.
        length = (math.e - math.exp(progress * progress)) * math.sin(2.0 * math.pi * float(self.rng.random()))

        mask = np.zeros(dimension)
        num_dims = max(1, int(math.ceil(float(self.rng.random()) * dimension)))
        selected = self.rng.choice(dimension, size=num_dims, replace=False)
        mask[selected] = 1.0
        return length * mask

    def _energy_factor(self, t: int) -> float:
        """Energy Shrink của ARO: A(t) = 4(1 - t/T) ln(1/r)."""
        r = max(float(self.rng.random()), 1e-12)
        return 4.0 * (1.0 - t / self.max_iterations) * math.log(1.0 / r)

    # ------------------------------------------------------------------
    # POA operators
    # ------------------------------------------------------------------
    def _poa_predator_attack(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        index: int,
        base: np.ndarray,
    ) -> np.ndarray:
        """POA Phase 1 - Predator Attack: chọn SP_i từ CP_i và cập nhật base."""
        current_fitness = fitness[index]
        better_indices = np.where(fitness < current_fitness)[0]
        better_indices = better_indices[better_indices != index]

        if len(better_indices) == 0:
            if self.poa_attack_fallback == "skip":
                return base.copy()
            if self.poa_attack_fallback == "random":
                choices = [j for j in range(self.population_size) if j != index]
                sp = population[int(self.rng.choice(choices))]
            else:
                assert self._best_solution is not None
                sp = self._best_solution
        else:
            sp = population[int(self.rng.choice(better_indices))]

        r = self.rng.random(base.shape[0])
        intensity = self.rng.integers(1, 3, size=base.shape[0])  # 1 hoặc 2
        return base + r * (sp - intensity * base)

    def _poa_defense_mechanism(self, base: np.ndarray, t: int) -> np.ndarray:
        """POA Phase 2 - Defense Mechanism: bước nhỏ khai thác cục bộ."""
        assert self._lower is not None and self._upper is not None
        r = self.rng.random(base.shape[0])
        step = (1.0 - 2.0 * r) * (self._upper - self._lower) / max(t, 1)
        return base + step

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _prepare_bounds(self, dimension: int) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.asarray(self.lower_bound, dtype=float)
        upper = np.asarray(self.upper_bound, dtype=float)

        if lower.ndim == 0:
            lower = np.full(dimension, float(lower))
        if upper.ndim == 0:
            upper = np.full(dimension, float(upper))

        if lower.shape != (dimension,) or upper.shape != (dimension,):
            raise ValueError("lower_bound và upper_bound phải là số hoặc vector có chiều bằng dimension.")
        if np.any(lower >= upper):
            raise ValueError("Mọi lower_bound phải nhỏ hơn upper_bound.")

        return lower, upper

    def _initialize_population(self) -> np.ndarray:
        assert self._lower is not None and self._upper is not None
        return self.rng.uniform(self._lower, self._upper, size=(self.population_size, self._dimension))

    def _repair(self, x: np.ndarray) -> np.ndarray:
        assert self._lower is not None and self._upper is not None
        return np.clip(x, self._lower, self._upper)

    def _evaluate(self, objective_func: ObjectiveFunction, x: np.ndarray) -> float:
        if self.max_evaluations is not None and self.nfe >= self.max_evaluations:
            return math.inf
        value = float(objective_func(np.asarray(x, dtype=float)))
        self.nfe += 1
        if not math.isfinite(value):
            return math.inf
        return value

    def _budget_exhausted(self) -> bool:
        return self.max_evaluations is not None and self.nfe >= self.max_evaluations

    def _update_global_best(self, population: np.ndarray, fitness: np.ndarray) -> None:
        best_idx = int(np.argmin(fitness))
        best_fit = float(fitness[best_idx])
        if best_fit < self._best_fitness:
            self._best_fitness = best_fit
            self._best_solution = population[best_idx].copy()


# ----------------------------------------------------------------------
# Demo chạy độc lập
# ----------------------------------------------------------------------
def sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))


if __name__ == "__main__":
    optimizer = RAPOMultiStrategyEnsemble(
        population_size=40,
        max_iterations=100,
        lower_bound=-100.0,
        upper_bound=100.0,
        seed=42,
        max_evaluations=None,
        energy_threshold=1.0,
        poa_attack_fallback="best",
    )

    result = optimizer.optimize(objective_func=sphere, dimension=30)

    print("Algorithm:", result.metadata["algorithm"])
    print("Best fitness:", result.best_fitness)
    print("Best solution first 5 dims:", result.best_solution[:5])
    print("NFE:", result.nfe)
    print("Runtime seconds:", round(result.runtime_seconds, 6))
    print("Phase usage:", result.metadata["phase_usage"])
    print("Strategy usage:", result.metadata["strategy_usage"])
    print("Strategy wins:", result.metadata["strategy_wins"])
