import numpy as np
import time
from typing import Dict, Any
from algorithms.base_optimizer import BaseOptimizer
from core.problem import Problem
from core.result import OptimizationResult
from core.evaluator import Evaluator
from core.boundary_handler import BoundaryHandler
from core.random_manager import RandomManager

class ARO(BaseOptimizer):
    name = "ARO"
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
        
        # Initialization
        population = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.array([evaluator.evaluate(ind) for ind in population])
        
        best_idx = np.argmin(fitness) if evaluator.minimization else np.argmax(fitness)
        best_solution = np.copy(population[best_idx])
        best_fitness = fitness[best_idx]
        
        convergence_curve = np.zeros(max_iter)
        
        for t in range(max_iter):
            # Cập nhật global best (trong evaluator đã tự cập nhật nhưng keep local tracking if needed)
            curr_best_idx = np.argmin(fitness) if evaluator.minimization else np.argmax(fitness)
            if evaluator.minimization:
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
                    
                r = np.random.rand()
                # Tính energy factor
                A = 4 * (1 - (t + 1) / max_iter) * np.log(1 / (r + 1e-10))
                
                # ARO logic (simplified standard)
                if A > 1: # Exploration / Detour Foraging
                    rand_idx = np.random.randint(0, pop_size)
                    while rand_idx == i:
                        rand_idx = np.random.randint(0, pop_size)
                    
                    R = np.random.randn(dim)
                    new_pos = population[rand_idx] + R * (population[i] - population[rand_idx])
                else: # Exploitation / Random Hiding
                    H = np.random.randn(dim)
                    new_pos = best_solution + A * H * (best_solution - population[i])
                    
                new_pos = BoundaryHandler.clip(new_pos, lb, ub)
                new_fitness = evaluator.evaluate(new_pos)
                
                # Greedy selection
                if evaluator.minimization:
                    if new_fitness < fitness[i]:
                        population[i] = new_pos
                        fitness[i] = new_fitness
                else:
                    if new_fitness > fitness[i]:
                        population[i] = new_pos
                        fitness[i] = new_fitness
                        
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
            metadata={"evaluations": evaluator.nfe}
        )
