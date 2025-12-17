"""
Base Policy - Базовый класс для всех политик управления
========================================================

Определяет интерфейс для различных стратегий управления:
- PDPolicy: PD контроллер
- TD3Policy: TD3 агент (Deep RL)
- CALFPolicy: CALF агент (будущее)
"""

from abc import ABC, abstractmethod
import numpy as np


class Policy(ABC):
    """
    Базовый класс для всех политик управления.

    Политика - это стратегия выбора действий на основе состояний.
    Может быть:
    - Классический контроллер (PD, LQR)
    - RL агент (TD3, SAC, CALF)
    - Адаптивный контроллер

    Все политики должны реализовать метод get_action().
    """

    @abstractmethod
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Получить действие для одного состояния.

        Parameters:
        -----------
        state : np.ndarray
            Состояние системы (например, [x, v] для 1D точки)

        Returns:
        --------
        action : np.ndarray
            Действие (управление) для системы
            Для 1D точки: scalar или array([u])
        """
        pass

    def get_actions_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Получить действия для батча состояний.

        По умолчанию - простой цикл, но можно переопределить
        для векторизации (например, для нейросетей).

        Parameters:
        -----------
        states : np.ndarray
            Массив состояний, shape (batch_size, state_dim)

        Returns:
        --------
        actions : np.ndarray
            Массив действий, shape (batch_size, action_dim)
        """
        return np.array([self.get_action(s) for s in states])

    def reset(self):
        """
        Сбросить внутреннее состояние политики (если есть).

        Некоторые политики могут иметь память (например, LSTM),
        этот метод позволяет сбросить её между эпизодами.

        По умолчанию - ничего не делает (stateless policy).
        """
        pass
