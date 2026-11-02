from dataclasses import dataclass

import numpy as np


@dataclass
class Source:
    """Точечный изотропный источник фотонов в начале координат.
    """

    Y: float = 10000

   
    def emit(self, E: float, seed: int | None = None) -> np.ndarray:
        """Разыграть начальные направления фотонов для события энергией E.
        """
        if E < 0:
            raise ValueError("E должен быть неотрицательным")

        n = int(round(self.Y * E))
        if n == 0:
            return np.empty((0, 3))

        rng = np.random.default_rng(seed)
        raw = rng.normal(size=(n, 3))
        
        return raw / np.linalg.norm(raw, axis=1, keepdims=True)