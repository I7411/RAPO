import numpy as np
import time
from typing import Dict, Any
from algorithms.base_optimizer import BaseOptimizer
from core.problem import Problem
from core.result import OptimizationResult
from core.evaluator import Evaluator
from core.boundary_handler import BoundaryHandler
from core.random_manager import RandomManager

class POA(BaseOptimizer):
    name = "POA"
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
        minimization = evaluator.minimization
        
        use_budget = config.get("use_evaluation_budget", False)
        max_evals = config.get("max_function_evaluations", 10000)
        
        start_time = time.perf_counter()
        
        population = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.array([evaluator.evaluate(ind) for ind in population])
        
        best_idx = np.argmin(fitness) if minimization else np.argmax(fitness)
        best_solution = np.copy(population[best_idx])
        best_fitness = fitness[best_idx]
        
        convergence_curve = np.zeros(max_iter)
        
        for t in range(max_iter):
            # Update best
            curr_best_idx = np.argmin(fitness) if minimization else np.argmax(fitness)
            if minimization:
                if fitness[curr_best_idx] < best_fitness:
                    best_fitness = fitness[curr_best_idx]
                    best_solution = np.copy(population[curr_best_idx])
            else:
                if fitness[curr_best_idx] > best_fitness:
                    best_fitness = fitness[curr_best_idx]
                    best_solution = np.copy(population[curr_best_idx])
                    
            for i in range(pop_size):
                if use_budget and evaluator.nfe >= max_evals:
                    break
                    
                # Phase 1: Predator Attack / Exploration
                if minimization:
                    better_indices = np.where(fitness < fitness[i])[0]
                else:
                    better_indices = np.where(fitness > fitness[i])[0]
                    
                if len(better_indices) > 0:
                    sp_idx = np.random.choice(better_indices)
                    sp = population[sp_idx]
                else:
                    sp = best_solution
                    
                r1 = np.random.rand()
                I = np.random.choice([1, 2])
                
                new_pos1 = population[i] + r1 * (sp - I * population[i])
                new_pos1 = BoundaryHandler.clip(new_pos1, lb, ub)
                new_fit1 = evaluator.evaluate(new_pos1)
                
                # Greedy selection phase 1
                if (minimization and new_fit1 < fitness[i]) or (not minimization and new_fit1 > fitness[i]):
                    population[i] = new_pos1
                    fitness[i] = new_fit1
                    
                # Phase 2: Defense Mechanism / Exploitation
                r2 = np.random.rand()
                step = (ub - lb) / (t + 1)
                new_pos2 = population[i] + (1 - 2 * r2) * step
                new_pos2 = BoundaryHandler.clip(new_pos2, lb, ub)
                new_fit2 = evaluator.evaluate(new_pos2)
                
                # Greedy selection phase 2
                if (minimization and new_fit2 < fitness[i]) or (not minimization and new_fit2 > fitness[i]):
                    population[i] = new_pos2
                    fitness[i] = new_fit2
                    
            convergence_curve[t] = best_fitness
            if use_budget and evaluator.nfe >= max_evals:
                break
                
        runtime = time.perf_counter() - start_time
        final_error = abs(best_fitness - problem.global_minimum) if problem.global_minimum is not None else None
        
        return OptimizationResult(
            algorithm=self.name,
            hybrid_type=self.hybrid_type,
            benchmark=problem.name,
            dimension=dim,
            run_id=config.get("run_id", 1),
            seed=seed,
            best_fitness=best_fitness,
            best_solution=best_solution.tolist(),
            convergence_curve=convergence_curve.tolist(),
            runtime_seconds=runtime,
            population_size=pop_size,
            max_iterations=max_iter,
            lower_bound=lb,
            upper_bound=ub,
            nfe=evaluator.nfe,
            final_error=final_error,
            metadata={
                "evaluations": evaluator.nfe,
                "algorithm_full_name": "Pufferfish Optimization Algorithm"
            }
        )
