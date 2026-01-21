# RL/base_env.py
"""
Абстрактный базовый класс для RL сред.

Определяет общий интерфейс для всех сред:
- state_dim, action_dim - размерности
- reset(), step() - основной API
- dynamics() - вычисление производной
- step_rk4() - RK4 интегрирование
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseEnv(ABC):
    """
    Абстрактный базовый класс для RL сред.
    
    Все среды должны наследоваться от этого класса и реализовать
    абстрактные методы: state_dim, action_dim, reset, step, dynamics.
    """
    
    def __init__(self, dt: float, max_action: float, goal_radius: float):
        """
        Инициализация базовой среды.
        
        Parameters:
        -----------
        dt : float
            Шаг интегрирования
        max_action : float
            Максимальное значение действия
        goal_radius : float
            Радиус целевой области
        """
        self.dt = dt
        self.max_action = max_action
        self.goal_radius = goal_radius
        self.state = None
        self.steps = 0
        self.max_steps = 5000
    
    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Размерность пространства состояний"""
        pass
    
    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Размерность пространства действий"""
        pass
    
    @abstractmethod
    def reset(self, state=None) -> np.ndarray:
        """
        Сбросить среду в начальное состояние.
        
        Parameters:
        -----------
        state : array-like, optional
            Начальное состояние. Если None, генерируется случайно.
            
        Returns:
        --------
        state : np.ndarray
            Начальное состояние
        """
        pass
    
    @abstractmethod
    def step(self, action) -> tuple:
        """
        Выполнить один шаг в среде.
        
        Parameters:
        -----------
        action : array-like
            Действие для выполнения
            
        Returns:
        --------
        next_state : np.ndarray
            Следующее состояние
        reward : float
            Награда
        done : bool
            Флаг завершения эпизода
        info : dict
            Дополнительная информация
        """
        pass
    
    @abstractmethod
    def dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Вычислить производную состояния (динамику системы).
        
        Parameters:
        -----------
        state : np.ndarray
            Текущее состояние
        action : np.ndarray
            Действие
            
        Returns:
        --------
        state_dot : np.ndarray
            Производная состояния
        """
        pass
    
    def distance_to_goal(self, state=None) -> float:
        """
        Расстояние до цели (по умолчанию - L2 норма состояния).
        
        Parameters:
        -----------
        state : np.ndarray, optional
            Состояние. Если None, используется текущее.
            
        Returns:
        --------
        distance : float
            Расстояние до цели
        """
        if state is None:
            state = self.state
        return np.linalg.norm(state)
    
    def step_rk4(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Один шаг интегрирования методом Рунге-Кутта 4-го порядка.
        
        Parameters:
        -----------
        state : np.ndarray
            Текущее состояние
        action : np.ndarray
            Действие
            
        Returns:
        --------
        next_state : np.ndarray
            Следующее состояние после интегрирования
        """
        k1 = self.dynamics(state, action)
        k2 = self.dynamics(state + self.dt * k1 / 2, action)
        k3 = self.dynamics(state + self.dt * k2 / 2, action)
        k4 = self.dynamics(state + self.dt * k3, action)
        return state + self.dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def render(self):
        """Вывести текущее состояние (для отладки)"""
        print(f"Step {self.steps}: state = {self.state}, distance = {self.distance_to_goal():.4f}")
