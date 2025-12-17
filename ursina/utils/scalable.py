from ursina import Entity
import numpy as np
from typing import Any

class Scalable(Entity):
    """
    Расширенный Entity с поддержкой масштабирования через ZoomManager
    """
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.real_position: np.ndarray = np.array(self.position)
        self.real_scale: np.ndarray = np.array(self.scale)

    def apply_transform(self, a: float, b: np.ndarray, **kwargs: Any) -> None:
        """Применить трансформацию масштабирования"""
        self.position = self.real_position * a + b
        self.scale = self.real_scale * a

    def __str__(self) -> str:
        return f'{self.position}'

    def __repr__(self) -> str:
        return f'{self.position}'
