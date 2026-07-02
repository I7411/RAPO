import numpy as np
import time
from typing import Dict, Any
from algorithms.base_optimizer import BaseOptimizer
from core.problem import Problem
from core.result import OptimizationResult
from core.evaluator import Evaluator
from core.boundary_handler import BoundaryHandler
from core.random_manager import RandomManager

class PSO(BaseOptimizer):
    name = "PSO"
    hybrid_type = ""

    def optimize(self, problem: Problem, config: Dict[str, Any], seed: int = None) -> OptimizationResult:
        if seed is not None:
            RandomManager.set_seed(seed)
            
        pop_size = config.get("population_size", 50)
        max_iter = config.get("max_iterations", 100)
        dim = problem.dimension
        lb = problem.lower_bound
        ub = problem.upper_bound
        
        # Params specific to PSO
        alg_cfg = config.get("PSO", {})
        c1 = alg_cfg.get("c1", 2.0)
        c2 = alg_cfg.get("c2", 2.0)
        w_max = alg_cfg.get("w_max", 0.9)
        w_min = alg_cfg.get("w_min", 0.4)
        
        evaluator = Evaluator(problem, config.get("minimization", True))
        evaluator.reset()
        minimization = evaluator.minimization
        
        use_budget = config.get("use_evaluation_budget", False)
        max_evals = config.get("max_function_evaluations", 10000)
        
        start_time = time.perf_counter()
        
        population = np.random.uniform(lb, ub, (pop_size, dim))
        velocity = np.zeros((pop_size, dim))
        
        pbest = np.copy(population)
        pbest_fit = np.array([evaluator.evaluate(ind) for ind in population])
        
        best_idx = np.argmin(pbest_fit) if minimization else np.argmax(pbest_fit)
        gbest = np.copy(population[best_idx])
        gbest_fit = pbest_fit[best_idx]
        
        convergence_curve = np.zeros(max_iter)
        
        for t in range(max_iter):
            w = w_max - t * ((w_max - w_min) / max_iter)
            
            for i in range(pop_size):
                if use_budget and evaluator.nfe >= max_evals:
                    break
                    
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)
                
                velocity[i] = w * velocity[i] + c1 * r1 * (pbest[i] - population[i]) + c2 * r2 * (gbest - population[i])
                population[i] = population[i] + velocity[i]
                
                population[i] = BoundaryHandler.clip(population[i], lb, ub)
                
                fit = evaluator.evaluate(population[i])
                
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
