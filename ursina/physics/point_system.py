"""
Класс системы точки
Система с одной координатой и одной скоростью
"""

import numpy as np
from typing import Optional
from .controllers.controller import Controller


class PointSystem:
    """
    Система точки с одной координатой и одной скоростью
    
    Состояние: [x, v] - координата и скорость
    Динамика: x_dot = v, v_dot = u (где u - управление)
    """
    
    def __init__(self, dt: float, initial_state: Optional[np.ndarray] = None, 
                 controller: Optional[Controller] = None, max_control: float = 5.0):
        """
        Инициализация системы точки
        
        Parameters:
        -----------
        dt : float
            Шаг интегрирования
        initial_state : np.ndarray, optional
            Начальное состояние [x, v]. Если None, используется [0, 0]
        controller : Controller, optional
            Контроллер для вычисления управления. Если None, управление = 0
        max_control : float, optional
            Максимальное абсолютное значение управления (по умолчанию 5.0)
        """
        self.dt = dt
        self.controller = controller
        self.max_control = max_control
        
        # Запоминаем начальное состояние навсегда
        if initial_state is None:
            self._initial_state = np.array([0.0, 0.0], dtype=np.float32)
        else:
            self._initial_state = np.array(initial_state, dtype=np.float32).copy()
        
        # Текущее состояние
        self.state = self._initial_state.copy()
    
    def get_state(self) -> np.ndarray:
        """
        Получить текущее состояние системы
        
        Returns:
        --------
        state : np.ndarray
            Текущее состояние [x, v]
        """
        return self.state.copy()
    
    def set_state(self, state: np.ndarray) -> None:
        """
        Установить состояние системы
        
        Parameters:
        -----------
        state : np.ndarray
            Новое состояние [x, v]
        """
        self.state = np.array(state, dtype=np.float32).copy()
    
    def reset_state(self) -> None:
        """
        Сбросить состояние в начальное
        """
        self.state = self._initial_state.copy()
    
    def generate_random_state(self) -> np.ndarray:
        """
        Сгенерировать случайное состояние
        
        Каждая координата равномерно распределена в диапазоне [-3, 3]
        
        Returns:
        --------
        state : np.ndarray
            Случайное состояние [x, v]
        """
        return np.random.uniform(-3.0, 3.0, size=2).astype(np.float32)
    
    def compute_derivative(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Вычислить производную системы (матрично)
        
        x_dot = v
        v_dot = u
        
        где u - управление от контроллера (или 0, если контроллера нет)
        
        Parameters:
        -----------
        state : np.ndarray, optional
            Состояние для вычисления производной. Если None, используется текущее состояние
            
        Returns:
        --------
        derivative : np.ndarray
            Производная [v, u]
        """
        if state is None:
            state = self.state
        
        x, v = state
        
        # Вычисляем управление
        if self.controller is not None:
            u = self.controller.get_control(state)
        else:
            # Если контроллера нет, используем self.u (для внешнего управления)
            u = self.u if self.u is not None else 0.0
        
        # Преобразуем u в скаляр, если это массив
        if isinstance(u, np.ndarray):
            u = float(u.item()) if u.size == 1 else float(u[0])
        else:
            u = float(u)
        
        # Ограничиваем управление
        u = np.clip(u, -self.max_control, self.max_control)
        
        # Производная: [x_dot, v_dot] = [v, u]
        return np.array([v, u], dtype=np.float32)
    
    def step(self) -> None:
        """
        Сделать один шаг интегрирования размером dt
        
        Используется метод Эйлера: state_new = state + dt * derivative
        """
        derivative = self.compute_derivative()
        self.state = self.state + derivative * self.dt
    
    def get_initial_state(self) -> np.ndarray:
        """
        Получить начальное состояние (которое было запомнено при создании)
        
        Returns:
        --------
        initial_state : np.ndarray
            Начальное состояние [x, v]
        """
        return self._initial_state.copy()