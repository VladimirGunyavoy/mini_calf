"""
Абстрактный класс контроллера
"""

from abc import ABC, abstractmethod
import numpy as np


class Controller(ABC):
    """
    Абстрактный класс контроллера
    
    Контроллер принимает полный стейт системы и возвращает управление
    """
    
    @abstractmethod
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
            Значение управления
        """
        pass