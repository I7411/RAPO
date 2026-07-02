import numpy as np
import time
from typing import Dict, Any
from algorithms.base_optimizer import BaseOptimizer
from core.problem import Problem
from core.result import OptimizationResult
from core.evaluator import Evaluator
from core.boundary_handler import BoundaryHandler
from core.random_manager import RandomManager

class GA(BaseOptimizer):
    name = "GA"
    hybrid_type = ""

    def optimize(self, problem: Problem, config: Dict[str, Any], seed: int = None) -> OptimizationResult:
        if seed is not None:
            RandomManager.set_seed(seed)
            
        pop_size = config.get("population_size", 50)
        max_iter = config.get("max_iterations", 100)
        dim = problem.dimension
        lb = problem.lower_bound
        ub = problem.upper_bound
        
        alg_cfg = config.get("GA", {})
        crossover_rate = alg_cfg.get("crossover_rate", 0.8)
        mutation_rate = alg_cfg.get("mutation_rate", 0.1)
        
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
            # Selection (Tournament)
            new_population = np.zeros_like(population)
            for i in range(pop_size):
                idx1, idx2 = np.random.randint(0, pop_size, 2)
                if minimization:
                    winner = idx1 if fitness[idx1] < fitness[idx2] else idx2
                else:
                    winner = idx1 if fitness[idx1] > fitness[idx2] else idx2
                new_population[i] = population[winner]
                
            # Crossover (Arithmetic)
            for i in range(0, pop_size - 1, 2):
                if np.random.rand() < crossover_rate:
                    alpha = np.random.rand()
                    p1, p2 = np.copy(new_population[i]), np.copy(new_population[i+1])
                    new_population[i] = alpha * p1 + (1 - alpha) * p2
                    new_population[i+1] = alpha * p2 + (1 - alpha) * p1
                    
            # Mutation (Gaussian)
            for i in range(pop_size):
                if np.random.rand() < mutation_rate:
                    new_population[i] += np.random.randn(dim) * (ub - lb) * 0.1
                    
            # Evaluation & Elitism
            new_fitness = np.zeros(pop_size)
            for i in range(pop_size):
                if use_budget and evaluator.nfe >= max_evals:
                    new_fitness[i] = float('inf') if minimization else float('-inf')
                    continue
                new_population[i] = BoundaryHandler.clip(new_population[i], lb, ub)
                new_fitness[i] = evaluator.evaluate(new_population[i])
            
            # Elitism: Giữ lại best solution
            worst_idx = np.argmax(new_fitness) if minimization else np.argmin(new_fitness)
            new_population[worst_idx] = np.copy(best_solution)
            new_fitness[worst_idx] = best_fitness
            
            population = new_population
            fitness = new_fitness
            
            curr_best_idx = np.argmin(fitness) if minimization else np.argmax(fitness)
            if minimization:
                if fitness[curr_best_idx] < best_fitness:
                    best_fitness = fitness[curr_best_idx]
                    best_solution = np.copy(population[curr_best_idx])
            else:
                if fitness[curr_best_idx] > best_fitness:
                    best_fitness = fitness[curr_best_idx]
                    best_solution = np.copy(population[curr_best_idx])
                    
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
