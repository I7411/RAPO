import numpy as np

class BoundaryHandler:
    @staticmethod
    def clip(solution: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
        """
        Xử lý nghiệm vượt biên bằng cách cắt (clip) nó trong khoảng [lower_bound, upper_bound].
        """
        return np.clip(solution, lower_bound, upper_bound)
