from experiments.run_single import run_single
from experiments.run_comparison import run_comparison

def verify():
    # 1. ARO, Sphere, D=5, Pop=10, Iter=10, Runs=3
    print(">>> VERIFICATION 1: ARO on Sphere")
    cfg1 = {
        "population_size": 10,
        "max_iterations": 10,
        "runs": 3
    }
    run_single("ARO", "Sphere", 5, cfg1)

    # 2. ARO, POA, RAPO-Energy-Switch, Rastrigin, D=10, Pop=20, Iter=30, Runs=5
    print("\n>>> VERIFICATION 2: Comparison on Rastrigin")
    cfg2 = {
        "population_size": 20,
        "max_iterations": 30,
        "runs": 5
    }
    algs = ["ARO", "POA", "RAPO_Energy_Switch"]
    run_comparison(algs, "Rastrigin", 10, cfg2)

if __name__ == "__main__":
    verify()
