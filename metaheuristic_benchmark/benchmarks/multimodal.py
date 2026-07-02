import numpy as np
from core.problem import Problem

def rastrigin(x: np.ndarray) -> float:
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def ackley(x: np.ndarray) -> float:
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e

def griewank(x: np.ndarray) -> float:
    sum_sq = np.sum(x**2) / 4000.0
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_sq - prod_cos + 1.0

def schwefel(x: np.ndarray) -> float:
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def levy(x: np.ndarray) -> float:
    w = 1 + (x - 1) / 4
    term1 = (np.sin(np.pi * w[0]))**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * (np.sin(np.pi * w[:-1] + 1))**2))
    term3 = (w[-1] - 1)**2 * (1 + (np.sin(2 * np.pi * w[-1]))**2)
    return term1 + term2 + term3

def rastrigin_problem(dim: int = 30) -> Problem:
    return Problem("Rastrigin", rastrigin, dim, -5.12, 5.12, 0.0, "multimodal")

def ackley_problem(dim: int = 30) -> Problem:
    return Problem("Ackley", ackley, dim, -32.0, 32.0, 0.0, "multimodal")

def griewank_problem(dim: int = 30) -> Problem:
    return Problem("Griewank", griewank, dim, -600.0, 600.0, 0.0, "multimodal")

def schwefel_problem(dim: int = 30) -> Problem:
    return Problem("Schwefel", schwefel, dim, -500.0, 500.0, 0.0, "multimodal")

def levy_problem(dim: int = 30) -> Problem:
    return Problem("Levy", levy, dim, -10.0, 10.0, 0.0, "multimodal")
