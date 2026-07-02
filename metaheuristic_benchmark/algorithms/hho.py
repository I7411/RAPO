import numpy as np
import time
import math
from typing import Dict, Any
from algorithms.base_optimizer import BaseOptimizer
from core.problem import Problem
from core.result import OptimizationResult
from core.evaluator import Evaluator
from core.boundary_handler import BoundaryHandler
from core.random_manager import RandomManager

class HHO(BaseOptimizer):
    name = "HHO"
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
        rabbit_energy = fitness[best_idx]
        rabbit_location = np.copy(population[best_idx])
        
        convergence_curve = np.zeros(max_iter)
        
        for t in range(max_iter):
            E1 = 2 * (1 - (t / max_iter))
            
            for i in range(pop_size):
                if use_budget and evaluator.nfe >= max_evals:
                    break
                    
                population[i] = BoundaryHandler.clip(population[i], lb, ub)
                fit = evaluator.evaluate(population[i])
                
                if (minimization and fit < rabbit_energy) or (not minimization and fit > rabbit_energy):
                    rabbit_energy = fit
                    rabbit_location = np.copy(population[i])
                    
                E0 = 2 * np.random.rand() - 1
                Escaping_Energy = E1 * E0
                
                if abs(Escaping_Energy) >= 1:
                    # Exploration phase
                    q = np.random.rand()
                    rand_hawk_idx = math.floor(pop_size * np.random.rand())
                    X_rand = population[rand_hawk_idx]
                    
                    if q < 0.5:
                        population[i] = X_rand - np.random.rand() * abs(X_rand - 2 * np.random.rand() * population[i])
                    else:
                        X_m = population.mean(axis=0)
                        population[i] = (rabbit_location - X_m) - np.random.rand() * (
                            lb + np.random.rand() * (ub - lb))
                else:
                    # Exploitation phase
                    r = np.random.rand()
                    if r >= 0.5 and abs(Escaping_Energy) < 0.5:
                        # Hard besiege
                        population[i] = rabbit_location - Escaping_Energy * abs(rabbit_location - population[i])
                    elif r >= 0.5 and abs(Escaping_Energy) >= 0.5:
                        # Soft besiege
                        J = 2 * (1 - np.random.rand())
                        population[i] = rabbit_location - population[i] - Escaping_Energy * abs(
                            J * rabbit_location - population[i])
                    elif r < 0.5 and abs(Escaping_Energy) >= 0.5:
                        # Soft besiege with progressive rapid dives
                        J = 2 * (1 - np.random.rand())
                        Y = rabbit_location - Escaping_Energy * abs(J * rabbit_location - population[i])
                        Y = BoundaryHandler.clip(Y, lb, ub)
                        
                        if evaluator.evaluate(Y) < fit:
                            population[i] = Y
                        else:
                            Z = Y + np.random.randn(dim) * 0.01 # simplified Levy
                            Z = BoundaryHandler.clip(Z, lb, ub)
                            if evaluator.evaluate(Z) < fit:
                                population[i] = Z
                    elif r < 0.5 and abs(Escaping_Energy) < 0.5:
                        # Hard besiege with progressive rapid dives
                        J = 2 * (1 - np.random.rand())
                        X_m = population.mean(axis=0)
                        Y = rabbit_location - Escaping_Energy * abs(J * rabbit_location - X_m)
                        Y = BoundaryHandler.clip(Y, lb, ub)
                        
                        if evaluator.evaluate(Y) < fit:
                            population[i] = Y
                        else:
                            Z = Y + np.random.randn(dim) * 0.01 # simplified Levy
                            Z = BoundaryHandler.clip(Z, lb, ub)
                            if evaluator.evaluate(Z) < fit:
                                population[i] = Z
                                
            convergence_curve[t] = rabbit_energy
            if use_budget and evaluator.nfe >= max_evals:
                break
                
        runtime = time.perf_counter() - start_time
        final_error = abs(rabbit_energy - problem.global_minimum) if problem.global_minimum is not None else None
        
        return OptimizationResult(
            algorithm=self.name,
            hybrid_type=self.hybrid_type,
            benchmark=problem.name,
            dimension=dim,
            run_id=config.get("run_id", 1),
            seed=seed,
            best_fitness=rabbit_energy,
            best_solution=rabbit_location.tolist(),
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
