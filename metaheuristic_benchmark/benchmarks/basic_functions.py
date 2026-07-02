import numpy as np
from core.problem import Problem

def sphere(x: np.ndarray) -> float:
    return np.sum(x**2)

def sphere_problem(dim: int = 30) -> Problem:
    return Problem(
        name="Sphere",
        function=sphere,
        dimension=dim,
        lower_bound=-100.0,
        upper_bound=100.0,
        global_minimum=0.0,
        category="unimodal"
    )
