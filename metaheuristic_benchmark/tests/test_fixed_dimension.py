import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from benchmarks.benchmark_registry import get_benchmark

def test_fixed_dimension():
    # 1. Booth D2 runs
    try:
        prob = get_benchmark("Booth", 2)
        assert prob.dimension == 2
        assert prob.is_fixed_dimension == True
    except ValueError:
        assert False, "Booth D2 should not raise ValueError"
        
    # 2. Booth D30 fails
    try:
        get_benchmark("Booth", 30)
        assert False, "Booth D30 should raise ValueError"
    except ValueError as e:
        assert "chỉ hỗ trợ dimension = 2" in str(e)
        
    # 3. Sphere D30 runs
    try:
        prob = get_benchmark("Sphere", 30)
        assert prob.dimension == 30
        assert prob.is_fixed_dimension == False
    except ValueError:
        assert False, "Sphere D30 should not raise ValueError"
        
    print("test_fixed_dimension passed")

if __name__ == "__main__":
    test_fixed_dimension()
