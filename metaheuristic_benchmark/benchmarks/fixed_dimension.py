import numpy as np
from core.problem import Problem

def booth(x: np.ndarray) -> float:
    if len(x) < 2: return 0.0
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

def matyas(x: np.ndarray) -> float:
    if len(x) < 2: return 0.0
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

def three_hump_camel(x: np.ndarray) -> float:
    if len(x) < 2: return 0.0
    return 2*x[0]**2 - 1.05*x[0]**4 + (x[0]**6)/6 + x[0]*x[1] + x[1]**2

def beale(x: np.ndarray) -> float:
    if len(x) < 2: return 0.0
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2

def easom(x: np.ndarray) -> float:
    if len(x) < 2: return 0.0
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi)**2 + (x[1] - np.pi)**2))

def booth_problem(dim: int = 2) -> Problem:
    return Problem("Booth", booth, 2, -10.0, 10.0, 0.0, "fixed_dimension", fixed_dimension=2, is_fixed_dimension=True)

def matyas_problem(dim: int = 2) -> Problem:
    return Problem("Matyas", matyas, 2, -10.0, 10.0, 0.0, "fixed_dimension", fixed_dimension=2, is_fixed_dimension=True)

def three_hump_camel_problem(dim: int = 2) -> Problem:
    return Problem("Three-Hump_Camel", three_hump_camel, 2, -5.0, 5.0, 0.0, "fixed_dimension", fixed_dimension=2, is_fixed_dimension=True)

def beale_problem(dim: int = 2) -> Problem:
    return Problem("Beale", beale, 2, -4.5, 4.5, 0.0, "fixed_dimension", fixed_dimension=2, is_fixed_dimension=True)

def easom_problem(dim: int = 2) -> Problem:
    return Problem("Easom", easom, 2, -100.0, 100.0, -1.0, "fixed_dimension", fixed_dimension=2, is_fixed_dimension=True)
