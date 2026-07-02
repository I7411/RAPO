import numpy as np
import time
from typing import Dict, Any
from algorithms.base_optimizer import BaseOptimizer
from core.problem import Problem
from core.result import OptimizationResult
from core.evaluator import Evaluator
from core.boundary_handler import BoundaryHandler
from core.random_manager import RandomManager

class GA_PSO(BaseOptimizer):
    name = "GA_PSO"
    hybrid_type = "basic"

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
        velocity = np.zeros((pop_size, dim))
        fitness = np.array([evaluator.evaluate(ind) for ind in population])
        
        pbest = np.copy(population)
        pbest_fit = np.copy(fitness)
        
        best_idx = np.argmin(fitness) if minimization else np.argmax(fitness)
        gbest = np.copy(population[best_idx])
        gbest_fit = fitness[best_idx]
        
        convergence_curve = np.zeros(max_iter)
        
        for t in range(max_iter):
            # Crossover GA
            new_population = np.copy(population)
            for i in range(0, pop_size - 1, 2):
                if np.random.rand() < 0.8:
                    alpha = np.random.rand()
                    p1, p2 = population[i], population[i+1]
                    new_population[i] = alpha * p1 + (1 - alpha) * p2
                    new_population[i+1] = alpha * p2 + (1 - alpha) * p1
            
            # PSO Update trên new_population
            w = 0.9 - t * (0.5 / max_iter)
            c1 = c2 = 2.0
            
            for i in range(pop_size):
                if use_budget and evaluator.nfe >= max_evals:
                    break
                    
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                
                velocity[i] = w * velocity[i] + c1 * r1 * (pbest[i] - new_population[i]) + c2 * r2 * (gbest - new_population[i])
                new_population[i] = new_population[i] + velocity[i]
                new_population[i] = BoundaryHandler.clip(new_population[i], lb, ub)
                
                # Mutation
                if np.random.rand() < 0.1:
                    new_population[i] += np.random.randn(dim) * (ub - lb) * 0.1
                    new_population[i] = BoundaryHandler.clip(new_population[i], lb, ub)
                
                fit = evaluator.evaluate(new_population[i])
                population[i] = new_population[i]
                
                if (minimization and fit < pbest_fit[i]) or (not minimization and fit > pbest_fit[i]):
                    pbest[i] = np.copy(population[i])
                    pbest_fit[i] = fit
                    
                    if (minimization and fit < gbest_fit) or (not minimization and fit > gbest_fit):
                        gbest = np.copy(population[i])
                        gbest_fit = fit
                        
            convergence_curve[t] = gbest_fit
            if use_budget and evaluator.nfe >= max_evals:
                break
                
        runtime = time.perf_counter() - start_time
        final_error = abs(gbest_fit - problem.global_minimum) if problem.global_minimum is not None else None
        
        return OptimizationResult(
            algorithm=self.name,
            hybrid_type=self.hybrid_type,
            benchmark=problem.name,
            dimension=dim,
            run_id=config.get("run_id", 1),
            seed=seed,
            best_fitness=gbest_fit,
            best_solution=gbest.tolist(),
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
