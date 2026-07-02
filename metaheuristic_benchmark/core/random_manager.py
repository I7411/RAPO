import numpy as np
import random

class RandomManager:
    @staticmethod
    def set_seed(seed: int):
        """
        Thiết lập seed cho numpy và Python random module để đảm bảo kết quả có thể lặp lại.
        """
        np.random.seed(seed)
        random.seed(seed)
