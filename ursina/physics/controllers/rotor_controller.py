"""
Контроллер ротора
Реализует управление u = -x
"""

import numpy as np
from .controller import Controller
import numpy as np


class RotorController(Controller):
    """
    Контроллер ротора
    
    Управление: u = -x
    где x - позиция системы (первый элемент стейта)
    """
    
    def get_control(self, state: np.ndarray) -> float:
        """
        Вычислить управление на основе состояния системы
        
        Parameters:
        -----------
        state : np.ndarray
            Текущее состояние системы [x, v]
            
        Returns:
        --------
        control : float
            Управление u = -x
        """
        x = state[0]
        v = state[1]
        return -x * 0.5 - 0.005 * v