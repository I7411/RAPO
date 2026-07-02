import numpy as np
from core.problem import Problem

# CEC2022 Global optimums based on the technical report
CEC2022_GLOBAL_MINIMUMS = {
    1: 300.0,
    2: 400.0,
    3: 600.0,
    4: 800.0,
    5: 900.0,
    6: 1800.0,
    7: 2000.0,
    8: 2200.0,
    9: 2300.0,
    10: 2400.0,
    11: 2600.0,
    12: 2700.0
}

def validate_cec2022_dimension(function_id: int, dimension: int):
    allowed_dimensions = {2, 10, 20}
    if dimension not in allowed_dimensions:
        raise ValueError("CEC2022 dimension must be one of: 2, 10, 20.")
    if dimension == 2 and function_id in {6, 7, 8}:
        raise ValueError(f"CEC2022_F{function_id} does not support D=2 (shuffle data not defined).")

def get_cec2022_bounds(dimension: int):
    return -100.0, 100.0

def get_cec2022_function(function_id: int, dimension: int):
    validate_cec2022_dimension(function_id, dimension)
    
    try:
        import opfunu
        import importlib
        cec2022_module = importlib.import_module("opfunu.cec_based.cec2022")
        func_class = getattr(cec2022_module, f"F{function_id}2022")
    except ImportError:
        raise ImportError("Cần cài đặt opfunu để chạy CEC2022. Hãy chạy: pip install opfunu")
    except AttributeError:
        raise AttributeError(f"opfunu không có hàm F{function_id}2022. Hãy kiểm tra lại phiên bản opfunu.")
    
    # Khởi tạo object của opfunu
    opfunu_obj = func_class(ndim=dimension)
    
    def objective(x):
        x_arr = np.asarray(x, dtype=float)
        
        if x_arr.ndim == 1:
            return float(opfunu_obj.evaluate(x_arr))
            
        if x_arr.ndim == 2:
            return np.apply_along_axis(opfunu_obj.evaluate, 1, x_arr).astype(float)
            
        raise ValueError("Input x must be 1D or 2D array.")
        
    objective.__name__ = f"CEC2022_F{function_id}"
    return objective

def get_all_cec2022_functions(dimension: int = 10):
    result = {}
    for function_id in range(1, 13):
        if dimension == 2 and function_id in {6, 7, 8}:
            continue
        result[f"CEC2022_F{function_id}"] = get_cec2022_function(function_id, dimension)
    return result

def get_cec2022_problem(name: str, dim: int) -> Problem:
    func_num = int(name.replace("CEC2022_F", ""))
    validate_cec2022_dimension(func_num, dim)
    
    global_min = CEC2022_GLOBAL_MINIMUMS.get(func_num, 0.0)
    
    objective_func = get_cec2022_function(func_num, dim)
    lb, ub = get_cec2022_bounds(dim)
    
    return Problem(
        name=name,
        function=objective_func,
        dimension=dim,
        lower_bound=lb,
        upper_bound=ub,
        global_minimum=global_min,
        category="cec2022",
        is_placeholder=False,
        is_official=True
    )
