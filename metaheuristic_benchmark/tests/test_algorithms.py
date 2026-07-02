import sys
import math
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from benchmarks.benchmark_registry import get_benchmark
from experiments.run_single import ALGORITHMS

def test_algorithms_basic_run():
    problem = get_benchmark("Sphere", dim=5)
    config = {
        "population_size": 10,
        "max_iterations": 5,
        "minimization": True
    }
    
    for name, alg_class in ALGORITHMS.items():
        try:
            alg = alg_class()
            res = alg.optimize(problem, config, seed=42)
            
            assert res is not None
            assert res.algorithm == alg.name
            assert not math.isnan(res.best_fitness)
            assert len(res.best_solution) == 5
            
            # Check bounds
            for val in res.best_solution:
                assert problem.lower_bound <= val <= problem.upper_bound
                
            print(f"Algorithm {name} passed basic run.")
        except Exception as e:
            print(f"Algorithm {name} failed: {e}")
            assert False, f"Algorithm {name} failed."

if __name__ == "__main__":
    test_algorithms_basic_run()
    print("All algorithm tests passed!")
