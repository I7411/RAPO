import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from experiments.run_single import run_single

def test_output_generation():
    # Set config override
    config = {
        "population_size": 10,
        "max_iterations": 5,
        "runs": 2
    }
    
    run_single("ARO", "Sphere", dimension=5, config_override=config)
    
    csv_dir = Path("outputs/csv")
    fig_dir = Path("outputs/figures")
    
    assert csv_dir.exists()
    assert fig_dir.exists()
    
    # Check if files are created (just basic check)
    csv_files = list(csv_dir.glob("*.csv"))
    fig_files = list(fig_dir.glob("*.png"))
    
    assert len(csv_files) > 0, "No CSV file was created."
    assert len(fig_files) > 0, "No PNG file was created."
    
    print("Output generation test passed!")

if __name__ == "__main__":
    test_output_generation()
