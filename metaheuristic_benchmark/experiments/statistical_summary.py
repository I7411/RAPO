import numpy as np
import datetime
from typing import List, Dict, Any
from core.result import OptimizationResult
from benchmarks.benchmark_registry import get_benchmark

def calculate_summary(results: List[OptimizationResult], config: dict = None) -> List[Dict[str, Any]]:
    if config is None:
        config = {}
    success_tolerance = config.get("success_tolerance", 1e-8)
    
    # Gom nhóm theo (algorithm, hybrid_type, benchmark, dimension)
    groups = {}
    for r in results:
        key = (r.algorithm, r.hybrid_type, r.benchmark, r.dimension)
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
        
    summaries = []
    
    for key, group in groups.items():
        alg_name, hybrid, bench_name, dim = key
        
        # Get problem metadata to check if official
        try:
            problem = get_benchmark(bench_name, dim)
            excluded_from_ranking = problem.is_placeholder or not problem.is_official
            bench_cat = problem.category
        except ValueError:
            excluded_from_ranking = False
            bench_cat = "unknown"
            
        fitnesses = [r.best_fitness for r in group]
        runtimes = [r.runtime_seconds for r in group]
        nfes = [r.nfe for r in group]
        
        # Errors
        final_errors = [r.final_error for r in group if r.final_error is not None]
        
        mean_fit = np.mean(fitnesses)
        best_fit = np.min(fitnesses)
        worst_fit = np.max(fitnesses)
        std_fit = np.std(fitnesses)
        median_fit = np.median(fitnesses)
        
        if final_errors:
            mean_err = np.mean(final_errors)
            best_err = np.min(final_errors)
            worst_err = np.max(final_errors)
            std_err = np.std(final_errors)
            success_count = sum(1 for e in final_errors if e <= success_tolerance)
            success_rate = success_count / len(final_errors)
        else:
            mean_err = best_err = worst_err = std_err = None
            success_count = success_rate = None
            
        nfe_mean = np.mean(nfes) if nfes else 0
        nfe_best = np.min(nfes) if nfes else 0
        nfe_worst = np.max(nfes) if nfes else 0
        
        avg_runtime = np.mean(runtimes)
        
        # Find alg full name from metadata if available
        alg_full_name = group[0].metadata.get("algorithm_full_name", alg_name)
        
        summaries.append({
            "algorithm": alg_name,
            "hybrid_type": hybrid,
            "algorithm_full_name": alg_full_name,
            "benchmark": bench_name,
            "benchmark_category": bench_cat,
            "dimension": dim,
            "runs": len(group),
            "mean": mean_fit,
            "best": best_fit,
            "worst": worst_fit,
            "std": std_fit,
            "median": median_fit,
            "mean_error": mean_err,
            "best_error": best_err,
            "worst_error": worst_err,
            "std_error": std_err,
            "nfe_mean": nfe_mean,
            "nfe_best": nfe_best,
            "nfe_worst": nfe_worst,
            "avg_runtime_seconds": avg_runtime,
            "rank_by_mean": 0,
            "rank_by_mean_error": 0,
            "success_count": success_count,
            "success_rate": success_rate,
            "excluded_from_ranking": excluded_from_ranking,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    # Tính rank (trên từng benchmark)
    benchmarks = set([s["benchmark"] for s in summaries])
    for b in benchmarks:
        b_summaries = [s for s in summaries if s["benchmark"] == b]
        
        # Lọc ra những cái hợp lệ để rank
        valid_for_ranking = [s for s in b_summaries if not s["excluded_from_ranking"]]
        
        # Rank by mean_error (if available)
        valid_for_ranking.sort(key=lambda x: (
            x["mean_error"] if x["mean_error"] is not None else float('inf'),
            x["best_error"] if x["best_error"] is not None else float('inf'),
            x["avg_runtime_seconds"]
        ))
        
        for i, s in enumerate(valid_for_ranking):
            s["rank_by_mean_error"] = i + 1 if s["mean_error"] is not None else None
            
        # Rank by mean
        valid_for_ranking.sort(key=lambda x: (
            x["mean"],
            x["best"],
            x["avg_runtime_seconds"]
        ))
        
        for i, s in enumerate(valid_for_ranking):
            s["rank_by_mean"] = i + 1
            
        # Các thuật toán bị loại khỏi ranking
        for s in b_summaries:
            if s["excluded_from_ranking"]:
                s["rank_by_mean"] = None
                s["rank_by_mean_error"] = None
            
    return summaries
