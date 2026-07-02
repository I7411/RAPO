import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from benchmarks.benchmark_registry import get_benchmark

def test_sphere():
    problem = get_benchmark("Sphere", dim=10)
    assert problem.function(np.zeros(10)) == 0.0

def test_rastrigin():
    problem = get_benchmark("Rastrigin", dim=10)
    assert problem.function(np.zeros(10)) == 0.0

def test_ackley():
    problem = get_benchmark("Ackley", dim=10)
    res = problem.function(np.zeros(10))
    assert abs(res) < 1e-5

if __name__ == "__main__":
    test_sphere()
    test_rastrigin()
    test_ackley()
    print("All benchmark tests passed!")
