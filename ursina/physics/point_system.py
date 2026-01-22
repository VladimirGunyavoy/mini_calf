# ursina/physics/point_system.py
"""
Класс системы точки.
Система с одной координатой и одной скоростью.

Состояние: [x, v] - координата и скорость
Динамика: x_dot = v, v_dot = u (где u - управление)
"""

import numpy as np
from typing import Optional

try:
    from .base_system import BaseDynamicalSystem
except ImportError:
    from base_system import BaseDynamicalSystem

try:
    from .controllers.controller import Controller
except ImportError:
    Controller = None  # Для запуска напрямую


class PointSystem(BaseDynamicalSystem):
    """
    Система точки с одной координатой и одной скоростью.
    
    Состояние: [x, v] - координата и скорость
    Динамика: x_dot = v, v_dot = u (где u - управление)
    """
    
    def __init__(self, dt: float, initial_state: Optional[np.ndarray] = None, 
                 controller: Optional['Controller'] = None, max_control: float = 5.0):
        """
        Инициализация системы точки.
        
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
        self.controller = controller
        self.max_control = max_control
        super().__init__(dt, initial_state)
    
    @property
    def state_dim(self) -> int:
        """Размерность состояния: [x, v]"""
        return 2
    
    def default_state(self) -> np.ndarray:
        """Состояние по умолчанию: [0, 0]"""
        return np.array([0.0, 0.0], dtype=np.float32)
    
    def generate_random_state(self) -> np.ndarray:
        """
        Сгенерировать случайное состояние.
        
        Каждая координата равномерно распределена в диапазоне [-3, 3]
        """
        return np.random.uniform(-3.0, 3.0, size=2).astype(np.float32)
    
    def state_to_position_3d(self, state: Optional[np.ndarray] = None) -> tuple:
        """
        Преобразовать состояние в 3D позицию для Ursina.
        
        Для 1D системы: x -> X, скорость не используется, Y=0, Z=0
        """
        if state is None:
            state = self.state
        x, v = state
        return (float(x), 0.0, 0.0)  # x -> X, остальные = 0
    
    def compute_derivative(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Вычислить производную системы.
        
        x_dot = v
        v_dot = u
        
        где u - управление от контроллера (или 0, если контроллера нет)
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


def test_point_system():
    """Тест системы точки"""
    system = PointSystem(dt=0.01)
    
    print(f"state_dim: {system.state_dim}")
    print(f"default_state: {system.default_state()}")
    print(f"initial_state: {system.get_initial_state()}")
    print(f"random_state: {system.generate_random_state()}")
    
    # Тест с начальным состоянием
    system = PointSystem(dt=0.01, initial_state=np.array([1.0, 0.5]))
    print(f"\nWith initial state [1, 0.5]:")
    print(f"  state: {system.get_state()}")
    print(f"  position_3d: {system.state_to_position_3d()}")
    
    # Тест интегрирования
    system.u = 1.0  # Ускорение = 1
    for i in range(10):
        system.step()
    print(f"  after 10 steps with u=1: {system.get_state()}")
    
    print("\nTest passed!")


if __name__ == "__main__":
    test_point_system()
