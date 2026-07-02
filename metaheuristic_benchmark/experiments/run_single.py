import json
from pathlib import Path
from algorithms.algorithm_registry import get_algorithm, list_algorithms
from benchmarks.benchmark_registry import get_benchmark
from utils.file_naming import generate_filename
from utils.csv_writer import write_detailed_csv
from utils.plotter import plot_convergence_curve
from utils.console import print_info, print_success, print_error
from utils.timer import measure_time
import datetime

def load_config(file_name: str) -> dict:
    with open(f"configs/{file_name}", "r") as f:
        return json.load(f)

def run_single(algorithm_name: str, benchmark_name: str, dimension: int, config_override: dict = None):
    print_info(f"Running {algorithm_name} on {benchmark_name} (D={dimension})")
    
    # Load configs
    default_config = load_config("default_config.json")
    algo_config = load_config("algorithm_config.json")
    
    config = default_config.copy()
    if algorithm_name in algo_config:
        config.update(algo_config[algorithm_name])
    
    # Extract base name for RAPO variants
    base_alg_name = algorithm_name.split("_")[0] if "RAPO" in algorithm_name else algorithm_name
    if base_alg_name in algo_config:
        config.update(algo_config[base_alg_name])
        
    if config_override:
        config.update(config_override)
        
    runs = config.get("runs", 1)
    seed_start = config.get("seed_start", 42)
    
    try:
        problem = get_benchmark(benchmark_name, dimension)
        if config.get("lower_bound") is not None:
            problem.lower_bound = float(config["lower_bound"])
        if config.get("upper_bound") is not None:
            problem.upper_bound = float(config["upper_bound"])
    except ValueError as e:
        print_error(str(e))
        return
    
    try:
        OptimizerClass = get_algorithm(algorithm_name)
    except ValueError as e:
        print_error(str(e))
        return
        
    optimizer = OptimizerClass()
    results = []
    
    seeds = []
    with measure_time() as t:
        for run_id in range(1, runs + 1):
            seed = seed_start + run_id
            seeds.append(seed)
            print_info(f"Run {run_id}/{runs} - Seed: {seed}")
            config["run_id"] = run_id
            res = optimizer.optimize(problem, config, seed)
            results.append(res)
            
    print_success(f"Finished {runs} runs in {t.duration:.2f}s")
    
    # Export CSV
    hybrid_type = results[0].hybrid_type
    csv_file = generate_filename("raw", optimizer.name, hybrid_type, benchmark_name, dimension, "csv")
    csv_path = Path("outputs/csv/raw") / csv_file
    write_detailed_csv(results, csv_path)
    
    # Plot Convergence Curve
    png_file = generate_filename("convergence", optimizer.name, hybrid_type, benchmark_name, dimension, "png")
    png_path = Path("outputs/figures/convergence") / png_file
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plot_convergence_curve(results, f"Convergence of {algorithm_name} on {benchmark_name}", png_path)
    
    # Export Metadata JSON
    meta_data = {
        "experiment_name": f"Single Run: {algorithm_name} on {benchmark_name}",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithms": [algorithm_name],
        "benchmark": benchmark_name,
        "dimension": dimension,
        "population_size": config.get("population_size", 50),
        "max_iterations": config.get("max_iterations", 100),
        "runs": runs,
        "seed_list": seeds,
        "use_evaluation_budget": config.get("use_evaluation_budget", False),
        "max_function_evaluations": config.get("max_function_evaluations", None),
        "success_tolerance": config.get("success_tolerance", 1e-8),
        "ranking_rule": "rank by mean_error if global_minimum exists, else rank by mean",
        "cec2017_placeholder_excluded": True
    }
    json_file = generate_filename("metadata", optimizer.name, hybrid_type, benchmark_name, dimension, "json")
    json_path = Path("outputs/metadata") / json_file
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=4)
    
    print_success(f"Results saved to:\n  {csv_path}\n  {png_path}\n  {json_path}")
    return results
