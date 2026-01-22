# ursina/physics/base_system.py
"""
Абстрактный базовый класс для физических систем.

Определяет общий интерфейс для всех систем в визуализации:
- state_dim - размерность состояния
- get_state(), set_state(), reset_state() - управление состоянием
- compute_derivative(), step() - динамика и интегрирование
- state_to_position_3d() - преобразование для Ursina
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class BaseDynamicalSystem(ABC):
    """
    Абстрактный базовый класс для физических систем.
    
    Все физические системы должны наследоваться от этого класса
    и реализовать абстрактные методы.
    """
    
    def __init__(self, dt: float, initial_state: Optional[np.ndarray] = None):
        """
        Инициализация базовой системы.
        
        Parameters:
        -----------
        dt : float
            Шаг интегрирования
        initial_state : np.ndarray, optional
            Начальное состояние. Если None, используется default_state()
        """
        self.dt = dt
        self._initial_state = initial_state if initial_state is not None else self.default_state()
        self._initial_state = np.array(self._initial_state, dtype=np.float32)
        self.state = self._initial_state.copy()
        self.u = None  # Внешнее управление
    
    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Размерность состояния"""
        pass
    
    @abstractmethod
    def default_state(self) -> np.ndarray:
        """Состояние по умолчанию"""
        pass
    
    @abstractmethod
    def compute_derivative(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Вычислить производную состояния.
        
        Parameters:
        -----------
        state : np.ndarray, optional
            Состояние. Если None, используется текущее.
            
        Returns:
        --------
        derivative : np.ndarray
            Производная состояния
        """
        pass
    
    @abstractmethod
    def generate_random_state(self) -> np.ndarray:
        """Сгенерировать случайное состояние"""
        pass
    
    @abstractmethod
    def state_to_position_3d(self, state: Optional[np.ndarray] = None) -> tuple:
        """
        Преобразовать состояние в 3D позицию для Ursina.
        
        Parameters:
        -----------
        state : np.ndarray, optional
            Состояние. Если None, используется текущее.
            
        Returns:
        --------
        position : tuple
            (x, y, z) координаты для Ursina
        """
        pass
    
    def get_state(self) -> np.ndarray:
        """Получить текущее состояние"""
        return self.state.copy()
    
    def set_state(self, state: np.ndarray) -> None:
        """Установить состояние"""
        self.state = np.array(state, dtype=np.float32).copy()
    
    def reset_state(self) -> None:
        """Сбросить состояние в начальное"""
        self.state = self._initial_state.copy()
    
    def get_initial_state(self) -> np.ndarray:
        """Получить начальное состояние"""
        return self._initial_state.copy()
    
    def step(self) -> None:
        """
        Сделать один шаг интегрирования (метод Эйлера).
        
        state_new = state + dt * derivative
        """
        derivative = self.compute_derivative()
        self.state = self.state + derivative * self.dt
