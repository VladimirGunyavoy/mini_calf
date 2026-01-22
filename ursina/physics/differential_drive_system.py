# ursina/physics/differential_drive_system.py
"""
Класс системы дифференциального привода (Unicycle model).

Состояние: [x, y, theta] - позиция и ориентация
Управление: [v, omega] - линейная и угловая скорость

Динамика:
    x_dot = v * cos(theta)
    y_dot = v * sin(theta)
    theta_dot = omega
"""

import numpy as np
from typing import Optional

try:
    from .base_system import BaseDynamicalSystem
except ImportError:
    from base_system import BaseDynamicalSystem


class DifferentialDriveSystem(BaseDynamicalSystem):
    """
    Дифференциальный привод (unicycle model) для визуализации.
    
    Состояние: [x, y, theta] - позиция + ориентация
    Управление: [v, omega] - линейная + угловая скорость
    """
    
    def __init__(self, dt: float, initial_state: Optional[np.ndarray] = None,
                 max_v: float = 1.0, max_omega: float = 2.0):
        """
        Инициализация системы дифференциального привода.
        
        Parameters:
        -----------
        dt : float
            Шаг интегрирования
        initial_state : np.ndarray, optional
            Начальное состояние [x, y, theta]. Если None, используется [0, 0, 0]
        max_v : float
            Максимальная линейная скорость
        max_omega : float
            Максимальная угловая скорость
        """
        self.max_v = max_v
        self.max_omega = max_omega
        super().__init__(dt, initial_state)
    
    @property
    def state_dim(self) -> int:
        """Размерность состояния: [x, y, theta]"""
        return 3
    
    def default_state(self) -> np.ndarray:
        """Состояние по умолчанию: [0, 0, 0]"""
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    def generate_random_state(self) -> np.ndarray:
        """
        Сгенерировать случайное состояние.
        
        x, y в [-2, 2], theta в [-pi, pi]
        """
        return np.array([
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-np.pi, np.pi),
        ], dtype=np.float32)
    
    def state_to_position_3d(self, state: Optional[np.ndarray] = None) -> tuple:
        """
        Преобразовать состояние в 3D позицию для Ursina.
        
        x -> X, y -> Z (Ursina использует Y для вертикали)
        """
        if state is None:
            state = self.state
        x, y, theta = state
        return (float(x), 0.0, float(y))  # x->X, y->Z
    
    def get_orientation(self, state: Optional[np.ndarray] = None) -> float:
        """
        Получить ориентацию (угол theta).
        
        Returns:
        --------
        theta : float
            Угол в радианах
        """
        if state is None:
            state = self.state
        return float(state[2])
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Нормализовать угол в [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def compute_derivative(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Вычислить производную системы.
        
        x_dot = v * cos(theta)
        y_dot = v * sin(theta)
        theta_dot = omega
        """
        if state is None:
            state = self.state
        x, y, theta = state
        
        # Получаем управление
        if self.u is not None:
            u = np.asarray(self.u, dtype=np.float32).flatten()
            v = np.clip(u[0], -self.max_v, self.max_v)
            omega = np.clip(u[1], -self.max_omega, self.max_omega)
        else:
            v, omega = 0.0, 0.0
        
        return np.array([
            v * np.cos(theta),
            v * np.sin(theta),
            omega
        ], dtype=np.float32)
    
    def step(self) -> None:
        """
        Сделать один шаг интегрирования с нормализацией угла.
        """
        derivative = self.compute_derivative()
        self.state = self.state + derivative * self.dt
        # Нормализация угла
        self.state[2] = self.normalize_angle(self.state[2])


def test_differential_drive_system():
    """Тест системы дифференциального привода"""
    system = DifferentialDriveSystem(dt=0.01)
    
    print(f"state_dim: {system.state_dim}")
    print(f"default_state: {system.default_state()}")
    print(f"initial_state: {system.get_initial_state()}")
    print(f"random_state: {system.generate_random_state()}")
    
    # Тест с начальным состоянием
    system = DifferentialDriveSystem(dt=0.01, initial_state=np.array([1.0, 0.5, 0.0]))
    print(f"\nWith initial state [1, 0.5, 0]:")
    print(f"  state: {system.get_state()}")
    print(f"  position_3d: {system.state_to_position_3d()}")
    print(f"  orientation: {np.degrees(system.get_orientation()):.1f} deg")
    
    # Тест движения вперёд
    system.u = [0.5, 0.0]  # Едем вперёд
    for i in range(100):
        system.step()
    print(f"  after 100 steps with v=0.5, omega=0: {system.get_state()}")
    
    # Тест поворота
    system.set_state(np.array([0.0, 0.0, 0.0]))
    system.u = [0.0, 1.0]  # Только поворачиваемся
    for i in range(100):
        system.step()
    print(f"  after 100 steps with v=0, omega=1: theta={np.degrees(system.get_orientation()):.1f} deg")
    
    # Тест нормализации угла
    system.set_state(np.array([0.0, 0.0, 3.0]))  # Почти pi
    system.u = [0.0, 1.0]
    for i in range(100):
        system.step()
    print(f"  angle normalization test: theta={system.get_orientation():.2f} rad (should be in [-pi, pi])")
    
    print("\nTest passed!")


if __name__ == "__main__":
    test_differential_drive_system()
