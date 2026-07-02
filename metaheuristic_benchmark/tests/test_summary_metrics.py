import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.result import OptimizationResult
from experiments.statistical_summary import calculate_summary
from benchmarks.benchmark_registry import get_benchmark

def test_summary_metrics():
    # Mock some results
    r1 = OptimizationResult(
        algorithm="MockAlg",
        hybrid_type="",
        benchmark="Sphere",
        dimension=2,
        run_id=1,
        seed=42,
        best_fitness=1e-9,
        best_solution=[0.0, 0.0],
        convergence_curve=[],
        runtime_seconds=1.0,
        population_size=10,
        max_iterations=10,
        lower_bound=-5.0,
        upper_bound=5.0,
        nfe=100,
        final_error=1e-9,
        metadata={"algorithm_full_name": "Mock Algorithm"}
    )
    
    r2 = OptimizationResult(
        algorithm="MockAlg",
        hybrid_type="",
        benchmark="Sphere",
        dimension=2,
        run_id=2,
        seed=43,
        best_fitness=1.0, # Failed run
        best_solution=[1.0, 0.0],
        convergence_curve=[],
        runtime_seconds=1.0,
        population_size=10,
        max_iterations=10,
        lower_bound=-5.0,
        upper_bound=5.0,
        nfe=100,
        final_error=1.0,
        metadata={"algorithm_full_name": "Mock Algorithm"}
    )
    
    config = {"success_tolerance": 1e-8}
    summary = calculate_summary([r1, r2], config)
    
    s = summary[0]
    assert s["algorithm"] == "MockAlg"
    assert s["algorithm_full_name"] == "Mock Algorithm"
    assert s["success_count"] == 1
    assert s["success_rate"] == 0.5
    assert s["mean_error"] == (1e-9 + 1.0) / 2
    assert s["best_error"] == 1e-9
    
    print("test_summary_metrics passed")

if __name__ == "__main__":
    test_summary_metrics()
