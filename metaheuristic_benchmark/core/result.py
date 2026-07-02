from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np

@dataclass
class OptimizationResult:
    algorithm: str
    hybrid_type: str
    benchmark: str
    dimension: int
    run_id: int
    seed: int
    best_fitness: float
    best_solution: List[float]
    convergence_curve: List[float]
    runtime_seconds: float
    population_size: int
    max_iterations: int
    lower_bound: float
    upper_bound: float
    nfe: int
    final_error: float | None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "hybrid_type": self.hybrid_type,
            "benchmark": self.benchmark,
            "dimension": self.dimension,
            "run_id": self.run_id,
            "seed": self.seed,
            "best_fitness": self.best_fitness,
            "best_solution": self.best_solution,
            "convergence_curve": self.convergence_curve,
            "runtime_seconds": self.runtime_seconds,
            "population_size": self.population_size,
            "max_iterations": self.max_iterations,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "nfe": self.nfe,
            "final_error": self.final_error,
            "metadata": self.metadata
        }
