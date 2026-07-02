import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.algorithm_registry import get_algorithm, list_algorithms, list_algorithms_by_type

def test_algorithm_registry():
    algs = list_algorithms()
    assert "ARO" in algs
    assert "POA" in algs
    assert "RAPO_Energy_Switch" in algs
    
    rapo_algs = list_algorithms_by_type("rapo")
    assert "RAPO_Energy_Switch" in rapo_algs
    assert "RAPO_Sequential" in rapo_algs
    assert "ARO" not in rapo_algs
    
    try:
        get_algorithm("RAPO_Energy_Switch")
    except ValueError:
        assert False, "Should find RAPO_Energy_Switch"
        
    try:
        get_algorithm("NonExistent")
        assert False, "Should raise ValueError for non-existent algorithm"
    except ValueError:
        pass
        
    print("test_algorithm_registry passed")

if __name__ == "__main__":
    test_algorithm_registry()
