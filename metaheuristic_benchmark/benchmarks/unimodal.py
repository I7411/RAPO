import numpy as np
from core.problem import Problem

def rosenbrock(x: np.ndarray) -> float:
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def zakharov(x: np.ndarray) -> float:
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
    return sum1 + sum2**2 + sum2**4

def bent_cigar(x: np.ndarray) -> float:
    return x[0]**2 + 1e6 * np.sum(x[1:]**2)

def sum_squares(x: np.ndarray) -> float:
    return np.sum(np.arange(1, len(x) + 1) * x**2)

def rosenbrock_problem(dim: int = 30) -> Problem:
    return Problem("Rosenbrock", rosenbrock, dim, -30.0, 30.0, 0.0, "unimodal")

def zakharov_problem(dim: int = 30) -> Problem:
    return Problem("Zakharov", zakharov, dim, -5.0, 10.0, 0.0, "unimodal")

def bent_cigar_problem(dim: int = 30) -> Problem:
    return Problem("Bent_Cigar", bent_cigar, dim, -100.0, 100.0, 0.0, "unimodal")

def sum_squares_problem(dim: int = 30) -> Problem:
    return Problem("Sum_Squares", sum_squares, dim, -10.0, 10.0, 0.0, "unimodal")
