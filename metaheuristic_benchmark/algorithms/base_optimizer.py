from abc import ABC, abstractmethod
from typing import Dict, Any
from core.problem import Problem
from core.result import OptimizationResult

class BaseOptimizer(ABC):
    name: str = "Base"
    hybrid_type: str = ""

    @abstractmethod
    def optimize(self, problem: Problem, config: Dict[str, Any], seed: int = None) -> OptimizationResult:
        """
        Thực thi tối ưu.
        
        Args:
            problem: Đối tượng Problem chứa hàm mục tiêu và các biên.
            config: Dictionary chứa cấu hình (population_size, max_iterations, etc.).
            seed: Seed dùng cho các module random.
            
        Returns:
            OptimizationResult: Kết quả quá trình tối ưu.
        """
        pass
