from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class Problem:
    name: str
    function: Callable[[Any], float]
    dimension: int
    lower_bound: float
    upper_bound: float
    global_minimum: float | None
    category: str
    fixed_dimension: int | None = None
    is_fixed_dimension: bool = False
    is_placeholder: bool = False
    is_official: bool = True
