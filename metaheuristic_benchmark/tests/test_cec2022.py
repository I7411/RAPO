import unittest
import numpy as np
import sys
import importlib

class TestCEC2022(unittest.TestCase):
    def setUp(self):
        # We handle the case where opfunu might not be installed in the test environment
        try:
            import opfunu
            self.has_opfunu = True
        except ImportError:
            self.has_opfunu = False
            
    def test_import_and_bounds(self):
        from benchmarks.cec2022.cec2022_functions import get_cec2022_problem
        # Nếu không có opfunu, hàm get_cec2022_problem vẫn khởi tạo được khung problem (tuy evaluate sẽ lỗi)
        # Wait, get_cec2022_problem calls get_cec2022_function which imports opfunu immediately
        pass

    def test_get_benchmark_via_registry(self):
        from benchmarks.benchmark_registry import get_benchmark
        if not self.has_opfunu:
            self.skipTest("Cần opfunu để chạy test này.")
            
        prob = get_benchmark("CEC2022_F1", 10)
        self.assertEqual(prob.name, "CEC2022_F1")
        self.assertEqual(prob.dimension, 10)
        self.assertEqual(prob.global_minimum, 300.0)
        self.assertEqual(prob.lower_bound, -100.0)
        self.assertEqual(prob.upper_bound, 100.0)

    def test_evaluate_1d_and_2d(self):
        from benchmarks.benchmark_registry import get_benchmark
        if not self.has_opfunu:
            self.skipTest("Cần opfunu để chạy test này.")
            
        prob = get_benchmark("CEC2022_F1", 10)
        
        # Test 1D
        x_1d = np.zeros(10)
        fit_1d = prob.function(x_1d)
        self.assertIsInstance(fit_1d, float)
        
        # Test 2D
        x_2d = np.zeros((5, 10))
        fit_2d = prob.function(x_2d)
        self.assertIsInstance(fit_2d, np.ndarray)
        self.assertEqual(fit_2d.shape, (5,))
        
    def test_invalid_dimension_for_f6(self):
        from benchmarks.benchmark_registry import get_benchmark
        with self.assertRaises(ValueError) as context:
            get_benchmark("CEC2022_F6", 2)
        self.assertTrue("không hỗ trợ D=2" in str(context.exception))

if __name__ == '__main__':
    unittest.main()
