"""
CALF Policy - Constrained Adaptive Learning Framework
======================================================

CALF агент с тремя режимами работы:
1. TD3 - основной режим (Deep RL агент)
2. Relax - промежуточный режим (смесь TD3 и Fallback)
3. Fallback - безопасный режим (PD контроллер)

Переключение между режимами основано на safety metric.
"""

import numpy as np
from .base_policy import Policy


class CALFPolicy(Policy):
    """
    CALF: TD3 с переключением на Relax/Fallback режимы.

    Три режима работы:
    - 'td3': Агент работает в нормальном режиме (высокий safety)
    - 'relax': Промежуточный режим - смесь TD3 и Fallback (средний safety)
    - 'fallback': Безопасный режим - PD контроллер (низкий safety)

    Переключение основано на safety_metric:
    - safety < fallback_threshold → fallback
    - fallback_threshold <= safety < relax_threshold → relax
    - safety >= relax_threshold → td3
    """

    MODE_TD3 = 'td3'
    MODE_RELAX = 'relax'
    MODE_FALLBACK = 'fallback'

    def __init__(
        self,
        td3_policy: Policy,
        pd_policy: Policy,
        fallback_threshold: float = 0.3,
        relax_threshold: float = 0.6,
        target: np.ndarray = None,
        dim: int = 1
    ):
        """
        Инициализация CALF политики.

        Parameters:
        -----------
        td3_policy : Policy
            TD3 агент (основная политика)
        pd_policy : Policy
            PD контроллер (fallback политика)
        fallback_threshold : float
            Порог для переключения в fallback режим
            safety < fallback_threshold → fallback
        relax_threshold : float
            Порог для переключения в relax режим
            safety < relax_threshold → relax
        target : np.ndarray, optional
            Целевая позиция для вычисления safety metric
        dim : int
            Размерность системы (1 для 1D, 2 для 2D)
        """
        self.td3 = td3_policy
        self.pd = pd_policy
        self.fallback_threshold = fallback_threshold
        self.relax_threshold = relax_threshold
        self.dim = dim

        # Целевая позиция для safety metric
        if target is None:
            self.target = np.zeros(dim)
        else:
            self.target = np.array(target)

        # Текущий режим (для визуализации)
        self.current_mode = self.MODE_TD3

        # Для batch обработки - массив режимов
        self.batch_modes = []

    def get_safety_metric(self, state: np.ndarray) -> float:
        """
        Вычислить метрику безопасности.

        Простая метрика на основе расстояния от цели:
        safety = 1 / (1 + distance)

        Чем ближе к цели, тем выше safety (можно использовать TD3).
        Чем дальше от цели, тем ниже safety (нужен fallback).

        Parameters:
        -----------
        state : np.ndarray
            Состояние системы [x, v] или [x, y, vx, vy]

        Returns:
        --------
        safety : float
            Метрика безопасности в диапазоне [0, 1]
        """
        # Извлекаем позицию
        if self.dim == 1:
            position = state[0:1]  # [x]
        else:
            position = state[:self.dim]  # [x, y, ...]

        # Расстояние от цели
        distance = np.linalg.norm(position - self.target)

        # Safety metric: чем ближе к цели, тем выше safety
        safety = 1.0 / (1.0 + distance)

        return safety

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Получить действие на основе текущего состояния.

        Переключается между режимами в зависимости от safety metric:
        - safety < fallback_threshold → fallback (PD)
        - fallback_threshold <= safety < relax_threshold → relax (mix)
        - safety >= relax_threshold → td3

        Parameters:
        -----------
        state : np.ndarray
            Состояние системы

        Returns:
        --------
        action : np.ndarray
            Действие (управление)
        """
        # Вычисляем метрику безопасности
        safety = self.get_safety_metric(state)

        # Выбираем режим и действие
        if safety < self.fallback_threshold:
            # Fallback режим - используем PD контроллер
            self.current_mode = self.MODE_FALLBACK
            action = self.pd.get_action(state)

        elif safety < self.relax_threshold:
            # Relax режим - смесь TD3 и PD
            self.current_mode = self.MODE_RELAX

            # Коэффициент смешивания (alpha)
            # При safety = fallback_threshold: alpha = 0 (только PD)
            # При safety = relax_threshold: alpha = 1 (только TD3)
            alpha = (safety - self.fallback_threshold) / \
                    (self.relax_threshold - self.fallback_threshold)

            # Смешанное действие
            td3_action = self.td3.get_action(state)
            pd_action = self.pd.get_action(state)
            action = alpha * td3_action + (1 - alpha) * pd_action

        else:
            # TD3 режим - используем агента
            self.current_mode = self.MODE_TD3
            action = self.td3.get_action(state)

        return action

    def get_actions_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Получить действия для батча состояний.

        Также сохраняет режимы для каждой среды в batch_modes.

        Parameters:
        -----------
        states : np.ndarray
            Массив состояний, shape (batch_size, state_dim)

        Returns:
        --------
        actions : np.ndarray
            Массив действий, shape (batch_size, action_dim)
        """
        # Вычисляем safety метрики для всех состояний
        safety_metrics = np.array([self.get_safety_metric(s) for s in states])

        # Определяем режимы для каждой среды
        self.batch_modes = []
        for safety in safety_metrics:
            if safety < self.fallback_threshold:
                self.batch_modes.append(self.MODE_FALLBACK)
            elif safety < self.relax_threshold:
                self.batch_modes.append(self.MODE_RELAX)
            else:
                self.batch_modes.append(self.MODE_TD3)

        # Получаем действия от обеих политик (для всех)
        td3_actions = self.td3.get_actions_batch(states)
        pd_actions = self.pd.get_actions_batch(states)

        # Комбинируем действия на основе режимов
        batch_size = len(states)
        action_dim = td3_actions.shape[1] if len(td3_actions.shape) > 1 else 1
        actions = np.zeros((batch_size, action_dim))

        for i, (safety, mode) in enumerate(zip(safety_metrics, self.batch_modes)):
            if mode == self.MODE_FALLBACK:
                # Только PD
                actions[i] = pd_actions[i]
            elif mode == self.MODE_RELAX:
                # Смесь TD3 и PD
                alpha = (safety - self.fallback_threshold) / \
                        (self.relax_threshold - self.fallback_threshold)
                actions[i] = alpha * td3_actions[i] + (1 - alpha) * pd_actions[i]
            else:  # MODE_TD3
                # Только TD3
                actions[i] = td3_actions[i]

        return actions

    def get_mode_for_env(self, env_idx: int) -> str:
        """
        Получить режим для конкретной среды (после batch обработки).

        Parameters:
        -----------
        env_idx : int
            Индекс среды в батче

        Returns:
        --------
        mode : str
            Режим: 'td3', 'relax', или 'fallback'
        """
        if env_idx < len(self.batch_modes):
            return self.batch_modes[env_idx]
        else:
            return self.current_mode

    def set_target(self, target: np.ndarray):
        """
        Установить новую целевую позицию.

        Parameters:
        -----------
        target : np.ndarray
            Новая целевая позиция
        """
        self.target = np.array(target)

    def set_thresholds(self, fallback_threshold: float = None, relax_threshold: float = None):
        """
        Обновить пороги переключения.

        Parameters:
        -----------
        fallback_threshold : float, optional
            Новый порог fallback
        relax_threshold : float, optional
            Новый порог relax
        """
        if fallback_threshold is not None:
            self.fallback_threshold = fallback_threshold
        if relax_threshold is not None:
            self.relax_threshold = relax_threshold

    def reset(self):
        """
        Сбросить внутреннее состояние политики.
        """
        self.current_mode = self.MODE_TD3
        self.batch_modes = []
        # Сброс вложенных политик
        self.td3.reset()
        self.pd.reset()







