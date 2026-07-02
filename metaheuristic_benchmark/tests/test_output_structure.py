import sys
import shutil
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from experiments.run_single import run_single

def test_output_structure():
    # Remove existing outputs if any for clean test
    out_dir = Path("outputs_test")
    if out_dir.exists():
        shutil.rmtree(out_dir)
        
    # We will override outputs path in run_single... wait, paths are hardcoded in run_single to 'outputs/'.
    # We can just run a single run with very low budget and check if the 'outputs/' files are created.
    
    config = {
        "runs": 1,
        "max_iterations": 2,
        "population_size": 5
    }
    
    run_single("GA", "Sphere", 2, config)
    
    # Check if raw csv exists
    raw_dir = Path("outputs/csv/raw")
    assert raw_dir.exists()
    
    # Check metadata exists
    meta_dir = Path("outputs/metadata")
    assert meta_dir.exists()
    
    print("test_output_structure passed")

if __name__ == "__main__":
    test_output_structure()
