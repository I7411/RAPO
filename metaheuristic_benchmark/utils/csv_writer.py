import csv
import json
from pathlib import Path
from typing import List, Dict, Any
from core.result import OptimizationResult

def ensure_dir(file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)

def write_detailed_csv(results: List[OptimizationResult], file_path: Path):
    ensure_dir(file_path)
    
    if not results:
        return
        
    fieldnames = [
        "algorithm", "hybrid_type", "benchmark", "dimension", "run_id", "seed",
        "best_fitness", "best_solution", "runtime_seconds",
        "population_size", "max_iterations", "lower_bound", "upper_bound", 
        "nfe", "final_error", "timestamp"
    ]
    
    with open(file_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            row = r.to_dict()
            # Convert list to string for CSV
            row["best_solution"] = json.dumps(row["best_solution"])
            # Remove convergence curve from detailed CSV to avoid bloat
            if "convergence_curve" in row:
                del row["convergence_curve"]
            if "metadata" in row:
                del row["metadata"]
                
            writer.writerow(row)

def write_summary_csv(summary_data: List[Dict[str, Any]], file_path: Path):
    ensure_dir(file_path)
    
    if not summary_data:
        return
        
    fieldnames = [
        "algorithm", "hybrid_type", "algorithm_full_name", "benchmark", "benchmark_category", "dimension", "runs",
        "mean", "best", "worst", "std", "median",
        "mean_error", "best_error", "worst_error", "std_error",
        "nfe_mean", "nfe_best", "nfe_worst",
        "avg_runtime_seconds", "rank_by_mean", "rank_by_mean_error",
        "success_count", "success_rate", "excluded_from_ranking", "timestamp"
    ]
    
    with open(file_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in summary_data:
            writer.writerow(row)
