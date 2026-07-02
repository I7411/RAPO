from pathlib import Path
from experiments.run_comparison import run_comparison
from experiments.statistical_summary import calculate_summary
from utils.csv_writer import write_summary_csv
from utils.plotter import plot_bar_comparison
from utils.file_naming import generate_overall_filename
from utils.console import print_info, print_success
import json
import datetime

def run_all_benchmarks(algorithms: list, benchmarks: list, dimension: int, config_override: dict = None):
    print_info("Starting full benchmark suite...")
    
    global_results = []
    
    for bench in benchmarks:
        res = run_comparison(algorithms, bench, dimension, config_override)
        if res:
            global_results.extend(res)
            
    if not global_results:
        return
        
    summary = calculate_summary(global_results)
    
    # Xuất summary CSV
    sum_csv_name = generate_overall_filename("summary", "all", dimension, "csv")
    sum_csv_path = Path("outputs/csv/summary") / sum_csv_name
    sum_csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_summary_csv(summary, sum_csv_path)
    
    # Bar chart cho trung bình rank (nếu có rank)
    bar_png_name = generate_overall_filename("bar_rank_by_mean_error", "all", dimension, "png")
    bar_png_path = Path("outputs/figures/bar") / bar_png_name
    bar_png_path.parent.mkdir(parents=True, exist_ok=True)
    plot_bar_comparison(summary, "rank_by_mean_error", "Average Rank Comparison", bar_png_path)
    
    # Export Metadata JSON
    meta_data = {
        "experiment_name": f"All Benchmarks: {', '.join(algorithms)} on {len(benchmarks)} functions",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithms": algorithms,
        "benchmarks": benchmarks,
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
    json_file = generate_overall_filename("metadata", "all", dimension, "json")
    json_path = Path("outputs/metadata") / json_file
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=4)
        
    print_success(f"Full suite finished. Global summary saved to {sum_csv_path}")
