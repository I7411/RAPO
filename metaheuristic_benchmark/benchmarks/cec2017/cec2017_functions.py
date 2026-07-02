import numpy as np
from core.problem import Problem
from benchmarks.cec2017.vendor.cec2017.functions import all_functions

def get_cec2017_function(function_id: int):
    if function_id < 1 or function_id > 30:
        raise ValueError("CEC2017 function_id must be from 1 to 30.")

    raw_func = all_functions[function_id - 1]

    def objective(x):
        x_arr = np.asarray(x, dtype=float)

        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
            return float(raw_func(x_arr)[0])

        if x_arr.ndim == 2:
            return raw_func(x_arr).astype(float)

        raise ValueError("Input x must be 1D or 2D array.")

    objective.__name__ = f"CEC2017_F{function_id}"
    return objective

def get_all_cec2017_functions(exclude_f2: bool = False):
    result = {}
    for function_id in range(1, 31):
        if exclude_f2 and function_id == 2:
            continue
        result[f"CEC2017_F{function_id}"] = get_cec2017_function(function_id)
    return result

def get_cec2017_bounds(dimension: int):
    return -100.0, 100.0

def validate_cec2017_dimension(dimension: int):
    allowed_dimensions = {10, 30, 50, 100}
    if dimension not in allowed_dimensions:
        raise ValueError("CEC2017 dimension must be one of: 10, 30, 50, 100.")

def get_cec2017_problem(name: str, dim: int) -> Problem:
    validate_cec2017_dimension(dim)
    func_num = int(name.replace("CEC2017_F", ""))
    
    # CEC2017 global minimum cho hàm thứ i thường là i * 100
    global_min = func_num * 100.0
    
    objective_func = get_cec2017_function(func_num)
    lb, ub = get_cec2017_bounds(dim)
    
    return Problem(
        name=name,
        function=objective_func,
        dimension=dim,
        lower_bound=lb,
        upper_bound=ub,
        global_minimum=global_min,
        category="cec2017",
        is_placeholder=False,
        is_official=True
    )
