"""
rapo_population_hybrid.py

RAPO Population-level Hybridization (RAPO-PLH)
================================================

Cơ chế lai:
    - Chia quần thể ban đầu thành 2 nhóm con: nhóm ARO và nhóm POA.
    - Nhóm ARO tiến hóa bằng cơ chế ARO:
        + Detour Foraging
        + Random Hiding
        + Energy Shrink để chọn pha ARO.
    - Nhóm POA tiến hóa bằng cơ chế POA-Pufferfish:
        + Phase 1: Predator Attack
        + Phase 2: Defense Mechanism
    - Sau mỗi M vòng lặp, trao đổi elite giữa hai nhóm:
        + Đưa X_best_ARO sang thay cá thể tệ nhất của nhóm POA.
        + Đưa X_best_POA sang thay cá thể tệ nhất của nhóm ARO.
    - Dùng greedy selection cho bài toán minimization.
    - Có đếm nfe để phục vụ so sánh công bằng.

File này được viết dạng standalone để có thể chạy thử trực tiếp.
Khi tích hợp vào framework thật, có thể thay _evaluate() bằng evaluator.evaluate().

Python: 3.11+
Dependency: numpy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import math
import time

import numpy as np


Array = np.ndarray
ObjectiveFunc = Callable[[Array], float]


@dataclass
class OptimizationResult:
    """Kết quả tối ưu hóa tối thiểu đủ dùng cho benchmark framework."""

    algorithm: str
    best_solution: Array
    best_fitness: float
    convergence_curve: List[float]
    nfe: int
    runtime_seconds: float
    metadata: Dict[str, object] = field(default_factory=dict)


class RAPOPopulationHybrid:
    """
    RAPO Population-level Hybridization / RAPO-PLH.

    Biến thể này giữ đồng thời 2 nhóm quần thể:
        - Nhóm ARO chạy theo cơ chế ARO.
        - Nhóm POA chạy theo cơ chế POA-Pufferfish.

    Sau mỗi `migration_interval` vòng lặp, nghiệm tốt nhất của mỗi nhóm được
    đưa sang nhóm còn lại bằng cơ chế best-to-worst replacement.

    Parameters
    ----------
    population_size:
        Tổng số cá thể N.
    max_iterations:
        Số vòng lặp tối đa T.
    lower_bound, upper_bound:
        Biên dưới và biên trên. Có thể là số hoặc vector numpy/list độ dài dimension.
    aro_ratio:
        Tỷ lệ cá thể thuộc nhóm ARO. Phần còn lại thuộc nhóm POA.
    migration_interval:
        M. Sau mỗi M vòng lặp sẽ trao đổi elite giữa hai nhóm.
    elite_count:
        Số cá thể elite trao đổi mỗi lần. Mặc định 1 để đúng sơ đồ cơ bản.
    seed:
        Random seed để tái lập kết quả.
    max_evaluations:
        Ngân sách đánh giá hàm mục tiêu. Nếu None thì chỉ dừng theo max_iterations.
    use_all_population_for_guidance:
        Nếu True, toán tử ARO/POA có thể tham chiếu toàn bộ quần thể.
        Nếu False, mỗi nhóm chỉ tham chiếu nội bộ nhóm của nó.
    """

    def __init__(
        self,
        population_size: int = 50,
        max_iterations: int = 100,
        lower_bound: float | List[float] | Array = -100.0,
        upper_bound: float | List[float] | Array = 100.0,
        aro_ratio: float = 0.5,
        migration_interval: int = 10,
        elite_count: int = 1,
        seed: Optional[int] = None,
        max_evaluations: Optional[int] = None,
        use_all_population_for_guidance: bool = False,
    ) -> None:
        if population_size < 4:
            raise ValueError("population_size phải >= 4 để chia được 2 nhóm có ý nghĩa.")
        if max_iterations < 1:
            raise ValueError("max_iterations phải >= 1.")
        if not 0.05 <= aro_ratio <= 0.95:
            raise ValueError("aro_ratio nên nằm trong [0.05, 0.95].")
        if migration_interval < 1:
            raise ValueError("migration_interval phải >= 1.")
        if elite_count < 1:
            raise ValueError("elite_count phải >= 1.")

        self.population_size = int(population_size)
        self.max_iterations = int(max_iterations)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.aro_ratio = float(aro_ratio)
        self.migration_interval = int(migration_interval)
        self.elite_count = int(elite_count)
        self.seed = seed
        self.max_evaluations = max_evaluations
        self.use_all_population_for_guidance = bool(use_all_population_for_guidance)

        self.rng = np.random.default_rng(seed)
        self.nfe = 0
        self._stop_by_budget = False

    def optimize(self, objective_func: ObjectiveFunc, dimension: int) -> OptimizationResult:
        """Chạy RAPO-PLH trên hàm mục tiêu minimization."""
        if dimension < 1:
            raise ValueError("dimension phải >= 1.")

        start_time = time.perf_counter()
        self.nfe = 0
        self._stop_by_budget = False

        lb, ub = self._prepare_bounds(dimension)
        population = self.rng.uniform(lb, ub, size=(self.population_size, dimension))
        fitness = self._evaluate_population(objective_func, population)

        aro_indices, poa_indices = self._split_population_indices()

        best_idx = int(np.argmin(fitness))
        best_solution = population[best_idx].copy()
        best_fitness = float(fitness[best_idx])

        convergence_curve: List[float] = [best_fitness]
        iteration_trace: List[Dict[str, object]] = []
        migration_events: List[Dict[str, object]] = []

        for t in range(1, self.max_iterations + 1):
            if self._budget_exhausted():
                break

            # 1) Nhóm ARO tiến hóa bằng cơ chế ARO.
            for idx in aro_indices:
                if self._budget_exhausted():
                    break

                if self.use_all_population_for_guidance:
                    guidance_population = population
                    guidance_fitness = fitness
                    guidance_indices = np.arange(self.population_size)
                else:
                    guidance_population = population[aro_indices]
                    guidance_fitness = fitness[aro_indices]
                    guidance_indices = aro_indices

                candidate = self._aro_candidate(
                    x_i=population[idx],
                    i_global=idx,
                    t=t,
                    population=guidance_population,
                    fitness=guidance_fitness,
                    global_indices=guidance_indices,
                    lb=lb,
                    ub=ub,
                )
                candidate = self._repair(candidate, lb, ub)
                candidate_fitness = self._evaluate(objective_func, candidate)

                if candidate_fitness <= fitness[idx]:
                    population[idx] = candidate
                    fitness[idx] = candidate_fitness

            # 2) Nhóm POA tiến hóa bằng cơ chế POA: Phase 1 rồi Phase 2.
            for idx in poa_indices:
                if self._budget_exhausted():
                    break

                if self.use_all_population_for_guidance:
                    guidance_population = population
                    guidance_fitness = fitness
                    guidance_indices = np.arange(self.population_size)
                else:
                    guidance_population = population[poa_indices]
                    guidance_fitness = fitness[poa_indices]
                    guidance_indices = poa_indices

                # POA Phase 1 - Predator Attack.
                candidate_p1 = self._poa_predator_attack_candidate(
                    x_i=population[idx],
                    f_i=fitness[idx],
                    i_global=idx,
                    population=guidance_population,
                    fitness=guidance_fitness,
                    global_indices=guidance_indices,
                    lb=lb,
                    ub=ub,
                )
                candidate_p1 = self._repair(candidate_p1, lb, ub)
                candidate_p1_fitness = self._evaluate(objective_func, candidate_p1)

                if candidate_p1_fitness <= fitness[idx]:
                    population[idx] = candidate_p1
                    fitness[idx] = candidate_p1_fitness

                if self._budget_exhausted():
                    break

                # POA Phase 2 - Defense Mechanism.
                candidate_p2 = self._poa_defense_candidate(population[idx], t, lb, ub)
                candidate_p2 = self._repair(candidate_p2, lb, ub)
                candidate_p2_fitness = self._evaluate(objective_func, candidate_p2)

                if candidate_p2_fitness <= fitness[idx]:
                    population[idx] = candidate_p2
                    fitness[idx] = candidate_p2_fitness

            # 3) Cập nhật nghiệm tốt nhất toàn cục.
            best_idx = int(np.argmin(fitness))
            if fitness[best_idx] < best_fitness:
                best_fitness = float(fitness[best_idx])
                best_solution = population[best_idx].copy()

            aro_best_idx, aro_best_fit = self._group_best(aro_indices, fitness)
            poa_best_idx, poa_best_fit = self._group_best(poa_indices, fitness)

            # 4) Trao đổi elite sau mỗi M vòng lặp.
            migrated = False
            if t % self.migration_interval == 0 and not self._budget_exhausted():
                event = self._migrate_elites(
                    population=population,
                    fitness=fitness,
                    aro_indices=aro_indices,
                    poa_indices=poa_indices,
                    t=t,
                )
                migration_events.append(event)
                migrated = True

                # Sau trao đổi elite, cập nhật lại best toàn cục.
                best_idx = int(np.argmin(fitness))
                if fitness[best_idx] < best_fitness:
                    best_fitness = float(fitness[best_idx])
                    best_solution = population[best_idx].copy()

            convergence_curve.append(best_fitness)
            iteration_trace.append(
                {
                    "iteration": t,
                    "best_fitness": best_fitness,
                    "aro_best_fitness": float(aro_best_fit),
                    "poa_best_fitness": float(poa_best_fit),
                    "aro_best_index": int(aro_best_idx),
                    "poa_best_index": int(poa_best_idx),
                    "migration": migrated,
                    "nfe": self.nfe,
                }
            )

            if self._budget_exhausted():
                break

        runtime = time.perf_counter() - start_time
        return OptimizationResult(
            algorithm="RAPO_Population_Hybrid",
            best_solution=best_solution,
            best_fitness=best_fitness,
            convergence_curve=convergence_curve,
            nfe=self.nfe,
            runtime_seconds=runtime,
            metadata={
                "population_size": self.population_size,
                "max_iterations": self.max_iterations,
                "dimension": dimension,
                "aro_ratio": self.aro_ratio,
                "aro_group_size": int(len(aro_indices)),
                "poa_group_size": int(len(poa_indices)),
                "migration_interval": self.migration_interval,
                "elite_count": self.elite_count,
                "seed": self.seed,
                "max_evaluations": self.max_evaluations,
                "stop_by_budget": self._stop_by_budget,
                "use_all_population_for_guidance": self.use_all_population_for_guidance,
                "migration_events": migration_events,
                "iteration_trace": iteration_trace,
            },
        )

    def _split_population_indices(self) -> Tuple[Array, Array]:
        """Chia quần thể thành nhóm ARO và nhóm POA."""
        indices = np.arange(self.population_size)
        self.rng.shuffle(indices)

        aro_size = int(round(self.population_size * self.aro_ratio))
        aro_size = max(1, min(self.population_size - 1, aro_size))

        aro_indices = np.sort(indices[:aro_size])
        poa_indices = np.sort(indices[aro_size:])
        return aro_indices, poa_indices

    def _aro_candidate(
        self,
        x_i: Array,
        i_global: int,
        t: int,
        population: Array,
        fitness: Array,
        global_indices: Array,
        lb: Array,
        ub: Array,
    ) -> Array:
        """Chọn pha ARO bằng Energy Shrink: A(t)>1 dùng Detour, ngược lại Hiding."""
        energy = self._aro_energy_factor(t)
        if energy > 1.0:
            return self._aro_detour_foraging_candidate(x_i, i_global, t, population, global_indices, lb, ub)
        return self._aro_random_hiding_candidate(x_i, t, lb, ub)

    def _aro_energy_factor(self, t: int) -> float:
        """A(t) = 4 * (1 - t/T) * ln(1/r), r in (0, 1)."""
        r = float(self.rng.uniform(1e-12, 1.0))
        return 4.0 * (1.0 - t / self.max_iterations) * math.log(1.0 / r)

    def _aro_running_operator(self, dimension: int, t: int) -> Array:
        """Running operator R = L * c trong ARO."""
        r2 = float(self.rng.random())
        r3 = float(self.rng.random())

        # L giảm dần về cuối, có dao động sin để tạo bước chạy ngẫu nhiên.
        L = (math.e - math.exp(((t - 1.0) / self.max_iterations) ** 2.0)) * math.sin(2.0 * math.pi * r2)

        c = np.zeros(dimension)
        number_of_mutated_dims = max(1, int(math.ceil(r3 * dimension)))
        selected_dims = self.rng.choice(dimension, size=number_of_mutated_dims, replace=False)
        c[selected_dims] = 1.0
        return L * c

    def _aro_detour_foraging_candidate(
        self,
        x_i: Array,
        i_global: int,
        t: int,
        population: Array,
        global_indices: Array,
        lb: Array,
        ub: Array,
    ) -> Array:
        """ARO Detour Foraging - pha khám phá."""
        dimension = x_i.size

        candidate_indices = np.where(global_indices != i_global)[0]
        if candidate_indices.size == 0:
            return x_i + self.rng.normal(0.0, 1.0, size=dimension) * 0.01 * (ub - lb)

        j_local = int(self.rng.choice(candidate_indices))
        x_j = population[j_local]

        R = self._aro_running_operator(dimension, t=t)
        r1 = float(self.rng.random())
        gaussian_noise = self.rng.normal(0.0, 1.0, size=dimension)
        perturb_flag = round(0.5 * (0.05 + r1))

        return x_j + R * (x_i - x_j) + perturb_flag * gaussian_noise

    def _aro_random_hiding_candidate(self, x_i: Array, t: int, lb: Array, ub: Array) -> Array:
        """ARO Random Hiding - pha khai thác."""
        dimension = x_i.size
        H = ((self.max_iterations - t + 1.0) / self.max_iterations) * float(self.rng.random())

        mask = np.zeros(dimension)
        selected_dim = int(self.rng.integers(0, dimension))
        mask[selected_dim] = 1.0

        burrow = x_i + H * mask * x_i
        R = self._aro_running_operator(dimension, t)
        r4 = float(self.rng.random())

        return x_i + R * (r4 * burrow - x_i)

    def _poa_predator_attack_candidate(
        self,
        x_i: Array,
        f_i: float,
        i_global: int,
        population: Array,
        fitness: Array,
        global_indices: Array,
        lb: Array,
        ub: Array,
    ) -> Array:
        """POA Phase 1 - Predator Attack towards Pufferfish."""
        better_local_indices = np.where((fitness < f_i) & (global_indices != i_global))[0]

        # Nếu không có cá thể tốt hơn trong nhóm, fallback về một bước nhiễu nhẹ quanh X_best nhóm.
        if better_local_indices.size == 0:
            best_local = int(np.argmin(fitness))
            x_best_group = population[best_local]
            step = self.rng.uniform(0.0, 1.0, size=x_i.size) * (x_best_group - x_i)
            noise = self.rng.normal(0.0, 0.01, size=x_i.size) * (ub - lb)
            return x_i + step + noise

        selected_local = int(self.rng.choice(better_local_indices))
        selected_pufferfish = population[selected_local]

        r = self.rng.uniform(0.0, 1.0, size=x_i.size)
        I = self.rng.integers(1, 3, size=x_i.size)  # I in {1, 2}
        return x_i + r * (selected_pufferfish - I * x_i)

    def _poa_defense_candidate(self, x_i: Array, t: int, lb: Array, ub: Array) -> Array:
        """POA Phase 2 - Defense Mechanism of Pufferfish."""
        r = self.rng.uniform(0.0, 1.0, size=x_i.size)
        denominator = max(1, t)
        return x_i + (1.0 - 2.0 * r) * ((ub - lb) / denominator)

    def _migrate_elites(
        self,
        population: Array,
        fitness: Array,
        aro_indices: Array,
        poa_indices: Array,
        t: int,
    ) -> Dict[str, object]:
        """Trao đổi elite giữa nhóm ARO và POA bằng best-to-worst replacement."""
        k = min(self.elite_count, len(aro_indices), len(poa_indices))

        aro_sorted = aro_indices[np.argsort(fitness[aro_indices])]
        poa_sorted = poa_indices[np.argsort(fitness[poa_indices])]

        aro_best_indices = aro_sorted[:k]
        aro_worst_indices = aro_sorted[-k:]
        poa_best_indices = poa_sorted[:k]
        poa_worst_indices = poa_sorted[-k:]

        aro_elites = population[aro_best_indices].copy()
        aro_elite_fitness = fitness[aro_best_indices].copy()
        poa_elites = population[poa_best_indices].copy()
        poa_elite_fitness = fitness[poa_best_indices].copy()

        # Đưa elite ARO sang thay worst POA.
        population[poa_worst_indices] = aro_elites
        fitness[poa_worst_indices] = aro_elite_fitness

        # Đưa elite POA sang thay worst ARO.
        population[aro_worst_indices] = poa_elites
        fitness[aro_worst_indices] = poa_elite_fitness

        return {
            "iteration": int(t),
            "elite_count": int(k),
            "aro_best_to_poa_worst": {
                "from_indices": aro_best_indices.astype(int).tolist(),
                "to_indices": poa_worst_indices.astype(int).tolist(),
                "fitness": aro_elite_fitness.astype(float).tolist(),
            },
            "poa_best_to_aro_worst": {
                "from_indices": poa_best_indices.astype(int).tolist(),
                "to_indices": aro_worst_indices.astype(int).tolist(),
                "fitness": poa_elite_fitness.astype(float).tolist(),
            },
        }

    def _group_best(self, group_indices: Array, fitness: Array) -> Tuple[int, float]:
        local_best = int(group_indices[int(np.argmin(fitness[group_indices]))])
        return local_best, float(fitness[local_best])

    def _prepare_bounds(self, dimension: int) -> Tuple[Array, Array]:
        lb = np.asarray(self.lower_bound, dtype=float)
        ub = np.asarray(self.upper_bound, dtype=float)

        if lb.ndim == 0:
            lb = np.full(dimension, float(lb))
        if ub.ndim == 0:
            ub = np.full(dimension, float(ub))

        if lb.shape != (dimension,) or ub.shape != (dimension,):
            raise ValueError("lower_bound và upper_bound phải là scalar hoặc vector độ dài dimension.")
        if np.any(lb >= ub):
            raise ValueError("Mỗi lower_bound phải nhỏ hơn upper_bound tương ứng.")

        return lb, ub

    @staticmethod
    def _repair(x: Array, lb: Array, ub: Array) -> Array:
        """Sửa biên/ràng buộc đơn giản cho benchmark liên tục."""
        return np.clip(x, lb, ub)

    def _evaluate(self, objective_func: ObjectiveFunc, x: Array) -> float:
        if self._budget_exhausted():
            self._stop_by_budget = True
            return float("inf")
        value = float(objective_func(x))
        self.nfe += 1
        return value

    def _evaluate_population(self, objective_func: ObjectiveFunc, population: Array) -> Array:
        values = np.empty(population.shape[0], dtype=float)
        for i, x in enumerate(population):
            values[i] = self._evaluate(objective_func, x)
        return values

    def _budget_exhausted(self) -> bool:
        if self.max_evaluations is None:
            return False
        exhausted = self.nfe >= self.max_evaluations
        if exhausted:
            self._stop_by_budget = True
        return exhausted


# Demo chạy độc lập.
def sphere(x: Array) -> float:
    return float(np.sum(x**2))


if __name__ == "__main__":
    optimizer = RAPOPopulationHybrid(
        population_size=50,
        max_iterations=100,
        lower_bound=-100.0,
        upper_bound=100.0,
        aro_ratio=0.5,
        migration_interval=10,
        elite_count=1,
        seed=42,
        max_evaluations=None,
        use_all_population_for_guidance=False,
    )

    result = optimizer.optimize(objective_func=sphere, dimension=30)

    print("Algorithm:", result.algorithm)
    print("Best fitness:", result.best_fitness)
    print("Best solution shape:", result.best_solution.shape)
    print("NFE:", result.nfe)
    print("Runtime seconds:", round(result.runtime_seconds, 6))
    print("Number of migrations:", len(result.metadata["migration_events"]))
