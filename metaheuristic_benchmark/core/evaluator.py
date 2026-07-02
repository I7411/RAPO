import numpy as np
from core.problem import Problem

class Evaluator:
    def __init__(self, problem: Problem, minimization: bool = True):
        self.problem = problem
        self.minimization = minimization
        self.nfe = 0
        
        # Track global bests
        self.global_best_fitness = float('inf') if minimization else float('-inf')
        self.global_best_solution = None

    def reset(self):
        self.nfe = 0
        self.global_best_fitness = float('inf') if self.minimization else float('-inf')
        self.global_best_solution = None

    def evaluate(self, x: np.ndarray) -> float:
        self.nfe += 1
        fitness = self.problem.function(x)
        
        if self.minimization:
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_solution = np.copy(x)
        else:
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_solution = np.copy(x)
                
        return fitness
