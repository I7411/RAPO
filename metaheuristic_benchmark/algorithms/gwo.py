import numpy as np
import time
from typing import Dict, Any
from algorithms.base_optimizer import BaseOptimizer
from core.problem import Problem
from core.result import OptimizationResult
from core.evaluator import Evaluator
from core.boundary_handler import BoundaryHandler
from core.random_manager import RandomManager

class GWO(BaseOptimizer):
    name = "GWO"
    hybrid_type = ""

    def optimize(self, problem: Problem, config: Dict[str, Any], seed: int = None) -> OptimizationResult:
        if seed is not None:
            RandomManager.set_seed(seed)
            
        pop_size = config.get("population_size", 50)
        max_iter = config.get("max_iterations", 100)
        dim = problem.dimension
        lb = problem.lower_bound
        ub = problem.upper_bound
        
        evaluator = Evaluator(problem, config.get("minimization", True))
        evaluator.reset()
        
        use_budget = config.get("use_evaluation_budget", False)
        max_evals = config.get("max_function_evaluations", 10000)
        
        start_time = time.perf_counter()
        
        population = np.random.uniform(lb, ub, (pop_size, dim))
        
        alpha_pos = np.zeros(dim)
        alpha_score = float("inf") if evaluator.minimization else float("-inf")
        
        beta_pos = np.zeros(dim)
        beta_score = float("inf") if evaluator.minimization else float("-inf")
        
        delta_pos = np.zeros(dim)
        delta_score = float("inf") if evaluator.minimization else float("-inf")
        
        convergence_curve = np.zeros(max_iter)
        
        for t in range(max_iter):
            for i in range(pop_size):
                if use_budget and evaluator.nfe >= max_evals:
                    break
                    
                population[i] = BoundaryHandler.clip(population[i], lb, ub)
                fitness = evaluator.evaluate(population[i])
                
                if evaluator.minimization:
                    if fitness < alpha_score:
                        delta_score, delta_pos = beta_score, np.copy(beta_pos)
                        beta_score, beta_pos = alpha_score, np.copy(alpha_pos)
                        alpha_score, alpha_pos = fitness, np.copy(population[i])
                    elif fitness < beta_score:
                        delta_score, delta_pos = beta_score, np.copy(beta_pos)
                        beta_score, beta_pos = fitness, np.copy(population[i])
                    elif fitness < delta_score:
                        delta_score, delta_pos = fitness, np.copy(population[i])
                else:
                    if fitness > alpha_score:
                        delta_score, delta_pos = beta_score, np.copy(beta_pos)
                        beta_score, beta_pos = alpha_score, np.copy(alpha_pos)
                        alpha_score, alpha_pos = fitness, np.copy(population[i])
                    elif fitness > beta_score:
                        delta_score, delta_pos = beta_score, np.copy(beta_pos)
                        beta_score, beta_pos = fitness, np.copy(population[i])
                    elif fitness > delta_score:
                        delta_score, delta_pos = fitness, np.copy(population[i])
            
            a = 2 - t * (2 / max_iter)
            
            for i in range(pop_size):
                for j in range(dim):
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha_pos[j] - population[i, j])
                    X1 = alpha_pos[j] - A1 * D_alpha
                    
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta_pos[j] - population[i, j])
                    X2 = beta_pos[j] - A2 * D_beta
                    
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta_pos[j] - population[i, j])
                    X3 = delta_pos[j] - A3 * D_delta
                    
                    population[i, j] = (X1 + X2 + X3) / 3.0
                    
            convergence_curve[t] = alpha_score
            if use_budget and evaluator.nfe >= max_evals:
                break
                
        runtime = time.perf_counter() - start_time
        final_error = abs(alpha_score - problem.global_minimum) if problem.global_minimum is not None else None
        
        return OptimizationResult(
            algorithm=self.name,
            hybrid_type=self.hybrid_type,
            benchmark=problem.name,
            dimension=dim,
            run_id=config.get("run_id", 1),
            seed=seed,
            best_fitness=alpha_score,
            best_solution=alpha_pos.tolist(),
            convergence_curve=convergence_curve.tolist(),
            runtime_seconds=runtime,
            population_size=pop_size,
            max_iterations=max_iter,
            lower_bound=lb,
            upper_bound=ub,
            nfe=evaluator.nfe,
            final_error=final_error,
            metadata={"evaluations": evaluator.nfe}
        )
