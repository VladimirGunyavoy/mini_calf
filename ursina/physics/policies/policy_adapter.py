"""
Policy Adapter - Адаптер для использования Policy как Controller
=================================================================

Позволяет использовать Policy (новый интерфейс) в старом коде,
который ожидает Controller.

Adapter pattern: Policy -> Controller interface
"""

import numpy as np
from ..controllers.controller import Controller
from .base_policy import Policy


class PolicyAdapter(Controller):
    """
    Адаптер для использования Policy как Controller.

    Старый код (PointSystem) использует Controller.get_control(state).
    Новый код (Phase 3+) использует Policy.get_action(state).

    PolicyAdapter позволяет использовать Policy там, где ожидается Controller.

    Example:
    --------
    >>> policy = PDPolicy(kp=1.0, kd=0.5)
    >>> controller = PolicyAdapter(policy)
    >>> point = PointSystem(dt=0.01, controller=controller)
    """

    def __init__(self, policy: Policy):
        """
        Инициализация адаптера.

        Parameters:
        -----------
        policy : Policy
            Политика для адаптации
        """
        self.policy = policy

    def get_control(self, state: np.ndarray) -> float:
        """
        Получить управление от политики.

        Адаптирует вызов Policy.get_action() к интерфейсу Controller.get_control().

        Parameters:
        -----------
        state : np.ndarray
            Состояние системы

        Returns:
        --------
        control : float
            Управление (скаляр для 1D системы)
        """
        # Получаем действие от политики
        action = self.policy.get_action(state)

        # Приводим к скаляру (для обратной совместимости с 1D системами)
        if isinstance(action, np.ndarray):
            if action.size == 1:
                return float(action[0])
            else:
                # Для multi-D систем возвращаем первую компоненту
                # TODO: В будущем нужно обобщить PointSystem на multi-D
                return float(action[0])
        else:
            return float(action)
