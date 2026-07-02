from typing import Dict, List
from core.problem import Problem

# Import all benchmarks
from benchmarks.basic_functions import sphere_problem
from benchmarks.unimodal import rosenbrock_problem, zakharov_problem, bent_cigar_problem, sum_squares_problem
from benchmarks.multimodal import rastrigin_problem, ackley_problem, griewank_problem, schwefel_problem, levy_problem
from benchmarks.fixed_dimension import booth_problem, matyas_problem, three_hump_camel_problem, beale_problem, easom_problem
from benchmarks.cec2017.cec2017_functions import get_cec2017_problem
from benchmarks.cec2022.cec2022_functions import get_cec2022_problem

_registry: Dict[str, callable] = {
    "Sphere": sphere_problem,
    "Rosenbrock": rosenbrock_problem,
    "Zakharov": zakharov_problem,
    "Bent_Cigar": bent_cigar_problem,
    "Sum_Squares": sum_squares_problem,
    "Rastrigin": rastrigin_problem,
    "Ackley": ackley_problem,
    "Griewank": griewank_problem,
    "Schwefel": schwefel_problem,
    "Levy": levy_problem,
    "Booth": booth_problem,
    "Matyas": matyas_problem,
    "Three-Hump_Camel": three_hump_camel_problem,
    "Beale": beale_problem,
    "Easom": easom_problem
}

def get_benchmark(name: str, dim: int = 30) -> Problem:
    """Lấy problem theo tên và số chiều."""
    if name.startswith("CEC2017_"):
        return get_cec2017_problem(name, dim)
        
    if name.startswith("CEC2022_"):
        return get_cec2022_problem(name, dim)
        
    if name not in _registry:
        raise ValueError(f"Benchmark '{name}' không tồn tại trong registry.")
        
    problem = _registry[name](dim)
    if problem.is_fixed_dimension and dim != problem.fixed_dimension:
        raise ValueError(f"Benchmark {name} là fixed-dimension benchmark và chỉ hỗ trợ dimension = {problem.fixed_dimension}. Vui lòng chọn D{problem.fixed_dimension}.")
        
    return problem

def list_benchmarks() -> List[str]:
    """Liệt kê tất cả tên benchmark cơ bản."""
    return list(_registry.keys())

def list_benchmarks_by_category(category: str, dim: int = 30) -> List[str]:
    """Liệt kê benchmark theo danh mục (unimodal, multimodal, fixed_dimension)."""
    return [name for name, func in _registry.items() if func(dim).category == category]
