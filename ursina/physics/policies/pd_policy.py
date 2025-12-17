"""
PD Policy - PD контроллер как политика
======================================

Пропорционально-дифференциальный контроллер.
Управление: u = Kp * error + Kd * error_dot

Где:
- error = target - position
- error_dot = -velocity (если target статичен)
"""

import numpy as np
from .base_policy import Policy


class PDPolicy(Policy):
    """
    PD контроллер как политика.

    Управление:
        u = Kp * (target - position) + Kd * (0 - velocity)
        u = Kp * error - Kd * velocity

    Для 1D точки:
        state = [x, v]
        action = Kp * (target_x - x) - Kd * v

    Для 2D точки:
        state = [x, y, vx, vy]
        action = [ux, uy]
        ux = Kp * (target_x - x) - Kd * vx
        uy = Kp * (target_y - y) - Kd * vy
    """

    def __init__(self, kp: float = 1.0, kd: float = 0.5, target: np.ndarray = None, dim: int = 1):
        """
        Инициализация PD политики.

        Parameters:
        -----------
        kp : float
            Коэффициент пропорциональности (position gain)
        kd : float
            Коэффициент дифференцирования (velocity gain)
        target : np.ndarray, optional
            Целевая позиция (по умолчанию - origin)
        dim : int
            Размерность управления (1 для 1D, 2 для 2D, etc.)
        """
        self.kp = kp
        self.kd = kd
        self.dim = dim

        # Целевая позиция (по умолчанию - начало координат)
        if target is None:
            self.target = np.zeros(dim)
        else:
            self.target = np.array(target)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Вычислить PD управление.

        Parameters:
        -----------
        state : np.ndarray
            Состояние системы
            Для 1D: [x, v]
            Для 2D: [x, y, vx, vy]

        Returns:
        --------
        action : np.ndarray
            Управление
            Для 1D: [u] или scalar
            Для 2D: [ux, uy]
        """
        # Извлекаем позицию и скорость
        if self.dim == 1:
            # 1D случай: state = [x, v]
            position = state[0:1]  # [x]
            velocity = state[1:2] if len(state) >= 2 else np.zeros(1)  # [v]
        else:
            # Multi-D случай: state = [x, y, ..., vx, vy, ...]
            position = state[:self.dim]  # [x, y, ...]
            if len(state) >= 2 * self.dim:
                velocity = state[self.dim:2*self.dim]  # [vx, vy, ...]
            else:
                velocity = np.zeros(self.dim)

        # PD управление: u = Kp * error - Kd * velocity
        error = self.target - position
        action = self.kp * error - self.kd * velocity

        return action

    def set_target(self, target: np.ndarray):
        """
        Установить новую целевую позицию.

        Parameters:
        -----------
        target : np.ndarray
            Новая целевая позиция
        """
        self.target = np.array(target)

    def set_gains(self, kp: float = None, kd: float = None):
        """
        Обновить коэффициенты PD контроллера.

        Parameters:
        -----------
        kp : float, optional
            Новый коэффициент пропорциональности
        kd : float, optional
            Новый коэффициент дифференцирования
        """
        if kp is not None:
            self.kp = kp
        if kd is not None:
            self.kd = kd
