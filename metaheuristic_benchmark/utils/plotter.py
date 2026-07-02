import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

def ensure_dir(file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)

def plot_convergence_curve(results: List[Any], title: str, file_path: Path):
    """Vẽ convergence curve trung bình của nhiều lần chạy cho một thuật toán hoặc so sánh"""
    ensure_dir(file_path)
    plt.figure(figsize=(10, 6))
    
    # Gom nhóm theo thuật toán
    alg_curves = {}
    for r in results:
        name = f"{r.algorithm}_{r.hybrid_type}" if r.hybrid_type else r.algorithm
        if name not in alg_curves:
            alg_curves[name] = []
        alg_curves[name].append(r.convergence_curve)
        
    for name, curves in alg_curves.items():
        # Lấy trung bình qua các run
        min_len = min(len(c) for c in curves)
        truncated_curves = [c[:min_len] for c in curves]
        mean_curve = np.mean(truncated_curves, axis=0)
        
        plt.plot(mean_curve, label=name, linewidth=2)
        
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (Log Scale)")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

def plot_boxplot(results: List[Any], title: str, file_path: Path):
    """Vẽ boxplot so sánh các thuật toán trên 1 benchmark"""
    ensure_dir(file_path)
    
    alg_fitness = {}
    for r in results:
        name = f"{r.algorithm}_{r.hybrid_type}" if r.hybrid_type else r.algorithm
        if name not in alg_fitness:
            alg_fitness[name] = []
        val = r.final_error if r.final_error is not None else r.best_fitness
        alg_fitness[name].append(val)
        
    names = list(alg_fitness.keys())
    data = [alg_fitness[n] for n in names]
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=names, patch_artist=True)
    plt.title(title)
    plt.ylabel("Final Error / Best Fitness (Log Scale)")
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

def plot_bar_comparison(summary_data: List[Dict[str, Any]], metric: str, title: str, file_path: Path):
    """Vẽ bar chart so sánh mean fitness hoặc rank"""
    ensure_dir(file_path)
    
    names = []
    values = []
    
    for row in summary_data:
        name = f"{row['algorithm']}_{row['hybrid_type']}" if row['hybrid_type'] else row['algorithm']
        names.append(name)
        values.append(row[metric])
        
    plt.figure(figsize=(10, 6))
    plt.bar(names, values, color='skyblue')
    plt.title(title)
    plt.ylabel(metric.replace("_", " ").title())
    if "fitness" in metric or "mean" in metric or "error" in metric:
        if "rank" not in metric and "rate" not in metric and "count" not in metric:
            plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()
