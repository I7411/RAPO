import unittest
import numpy as np
from benchmarks.cec2017.cec2017_functions import get_cec2017_problem, get_cec2017_function
from benchmarks.benchmark_registry import get_benchmark

class TestCEC2017(unittest.TestCase):
    def test_import_and_bounds(self):
        problem = get_cec2017_problem("CEC2017_F1", 10)
        self.assertEqual(problem.name, "CEC2017_F1")
        self.assertEqual(problem.dimension, 10)
        self.assertEqual(problem.lower_bound, -100.0)
        self.assertEqual(problem.upper_bound, 100.0)
        self.assertEqual(problem.global_minimum, 100.0)
        
    def test_evaluate_1d(self):
        problem = get_benchmark("CEC2017_F1", 10)
        x = np.zeros(10)
        fitness = problem.function(x)
        self.assertIsInstance(fitness, float)
        # Tại gốc toạ độ x=0, hàm CEC có thể có giá trị cụ thể, nhưng ta chỉ test nó chạy được và trả về float
        self.assertTrue(fitness > 0)
        
    def test_evaluate_2d(self):
        problem = get_benchmark("CEC2017_F1", 10)
        x = np.zeros((5, 10))
        fitness = problem.function(x)
        self.assertIsInstance(fitness, np.ndarray)
        self.assertEqual(fitness.shape, (5,))

if __name__ == '__main__':
    unittest.main()
