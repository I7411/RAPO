from pathlib import Path
from experiments.run_single import run_single
from experiments.statistical_summary import calculate_summary
from utils.csv_writer import write_summary_csv
from utils.plotter import plot_boxplot
from utils.file_naming import generate_summary_filename
from utils.console import print_info, print_success
import json
import datetime

def run_comparison(algorithms: list, benchmark_name: str, dimension: int, config_override: dict = None):
    print_info(f"Starting comparison on {benchmark_name} (D={dimension})")
    
    all_results = []
    for alg in algorithms:
        res = run_single(alg, benchmark_name, dimension, config_override)
        if res:
            all_results.extend(res)
            
    if not all_results:
        return
        
    # Tính summary
    summary = calculate_summary(all_results)
    
    # Xuất summary CSV
    sum_csv_name = generate_summary_filename("summary", benchmark_name, dimension, "csv")
    sum_csv_path = Path("outputs/csv/summary") / sum_csv_name
    sum_csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_summary_csv(summary, sum_csv_path)
    
    # Vẽ boxplot
    box_png_name = generate_summary_filename("boxplot", benchmark_name, dimension, "png")
    box_png_path = Path("outputs/figures/boxplot") / box_png_name
    box_png_path.parent.mkdir(parents=True, exist_ok=True)
    plot_boxplot(all_results, f"Boxplot Comparison on {benchmark_name}", box_png_path)
    
    # Export Metadata JSON
    meta_data = {
        "experiment_name": f"Comparison: {', '.join(algorithms)} on {benchmark_name}",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithms": algorithms,
        "benchmark": benchmark_name,
        "dimension": dimension,
        "population_size": config_override.get("population_size", 50) if config_override else 50,
        "max_iterations": config_override.get("max_iterations", 100) if config_override else 100,
        "runs": config_override.get("runs", 1) if config_override else 1,
        "use_evaluation_budget": config_override.get("use_evaluation_budget", False) if config_override else False,
        "max_function_evaluations": config_override.get("max_function_evaluations", None) if config_override else None,
        "success_tolerance": config_override.get("success_tolerance", 1e-8) if config_override else 1e-8,
        "ranking_rule": "rank by mean_error if global_minimum exists, else rank by mean",
        "cec2017_placeholder_excluded": True
    }
    json_file = generate_summary_filename("metadata", benchmark_name, dimension, "json")
    json_path = Path("outputs/metadata") / json_file
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=4)
    
    print_success(f"Comparison finished. Summary saved to {sum_csv_path}")
    return all_results
