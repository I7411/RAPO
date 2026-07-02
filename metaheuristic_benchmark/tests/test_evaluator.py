import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.problem import Problem
from core.evaluator import Evaluator

def test_evaluator_nfe():
    def dummy_func(x):
        return sum(x**2)
        
    problem = Problem("Dummy", dummy_func, 2, -5.0, 5.0, 0.0, "test")
    evaluator = Evaluator(problem, minimization=True)
    
    assert evaluator.nfe == 0
    
    x1 = np.array([1.0, 1.0])
    evaluator.evaluate(x1)
    assert evaluator.nfe == 1
    
    # Evaluate 10 times
    for _ in range(10):
        evaluator.evaluate(x1)
        
    assert evaluator.nfe == 11
    
    evaluator.reset()
    assert evaluator.nfe == 0
    
    print("test_evaluator_nfe passed")

if __name__ == "__main__":
    test_evaluator_nfe()
