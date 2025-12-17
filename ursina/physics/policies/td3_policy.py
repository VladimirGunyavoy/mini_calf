"""
TD3 Policy - Twin Delayed Deep Deterministic Policy Gradient
=============================================================

TD3 агент для Deep RL.
Использует обученного агента или случайные действия (stub).

Phase 9: Интеграция реального TD3 агента с PyTorch.
"""

import sys
from pathlib import Path
import numpy as np

from .base_policy import Policy

# Опциональный импорт PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not found. TD3Policy will work only in stub mode.")

# Добавляем путь к RL модулю
RL_PATH = Path(__file__).parent.parent.parent.parent / "RL"
if str(RL_PATH) not in sys.path:
    sys.path.insert(0, str(RL_PATH))

try:
    if TORCH_AVAILABLE:
        from td3 import TD3
        TD3_AVAILABLE = True
    else:
        TD3_AVAILABLE = False
except ImportError:
    TD3_AVAILABLE = False
    print("[WARNING] TD3 module not found. Using stub mode only.")


class TD3Policy(Policy):
    """
    TD3 агент (Twin Delayed DDPG).

    Может работать в двух режимах:
    1. Stub mode (agent=None): случайные действия для тестирования
    2. Real mode (agent!=None): обученная нейросеть для inference

    Архитектура TD3:
    - Actor: state -> action
    - Critic 1: (state, action) -> Q-value
    - Critic 2: (state, action) -> Q-value (twin)
    - Target networks для стабильности
    """

    def __init__(
        self,
        agent=None,
        action_dim: int = 1,
        action_scale: float = 1.0,
        device: str = None
    ):
        """
        Инициализация TD3 политики.

        Parameters:
        -----------
        agent : TD3, optional
            TD3 агент с обученными весами
            Если None - используется случайная политика (stub)
        action_dim : int
            Размерность действия (1 для 1D, 2 для 2D, etc.)
        action_scale : float
            Масштаб случайных действий (для stub режима)
        device : str, optional
            Устройство для PyTorch ('cpu' или 'cuda')
            Если None - автоматический выбор
        """
        self.agent = agent
        self.action_dim = action_dim
        self.action_scale = action_scale

        # Выбор устройства
        if device is None and TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif TORCH_AVAILABLE:
            self.device = torch.device(device)
        else:
            self.device = "cpu"  # Fallback if torch not available

        # Переводим агента в eval mode если он существует
        if self.agent is not None:
            self.agent.actor.eval()

        mode = "stub" if agent is None else "real TD3"
        print(f"[OK] TD3Policy initialized ({mode}, device={self.device})")

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Получить действие от TD3 агента.

        Parameters:
        -----------
        state : np.ndarray
            Состояние системы

        Returns:
        --------
        action : np.ndarray
            Действие от агента
            Для stub: случайное действие ~ N(0, action_scale)
            Для реального агента: действие из нейросети
        """
        if self.agent is None:
            # Stub режим: случайные действия
            action = np.random.normal(0, self.action_scale, self.action_dim)
            return action
        else:
            # Реальный агент (Phase 9)
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch is required for real TD3 agent. Install torch.")

            with torch.no_grad():
                state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
                action_tensor = self.agent.actor(state_tensor)
                action = action_tensor.cpu().numpy().flatten()
            return action

    def get_actions_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Получить действия для батча состояний (batch inference).

        Parameters:
        -----------
        states : np.ndarray
            Батч состояний размера (n_envs, state_dim)

        Returns:
        --------
        actions : np.ndarray
            Батч действий размера (n_envs, action_dim)
        """
        if self.agent is None:
            # Stub режим: случайные действия для батча
            n_envs = states.shape[0]
            actions = np.random.normal(0, self.action_scale, (n_envs, self.action_dim))
            return actions
        else:
            # Реальный агент - batch inference
            with torch.no_grad():
                states_tensor = torch.FloatTensor(states).to(self.device)
                actions_tensor = self.agent.actor(states_tensor)
                actions = actions_tensor.cpu().numpy()
            return actions

    def load_weights(self, path: str):
        """
        Загрузить веса обученного агента.

        Parameters:
        -----------
        path : str
            Путь к файлу с весами (.pth)
        """
        if self.agent is None:
            raise ValueError("Cannot load weights: agent is None. Create TD3 agent first.")

        self.agent.load(path)
        self.agent.actor.eval()  # Переводим в eval mode после загрузки
        print(f"[OK] TD3 weights loaded from {path}")

    def save_weights(self, path: str):
        """
        Сохранить веса агента.

        Parameters:
        -----------
        path : str
            Путь для сохранения весов
        """
        if self.agent is None:
            raise ValueError("Cannot save weights: agent is None")

        self.agent.save(path)
        print(f"[OK] TD3 weights saved to {path}")

    def train_mode(self):
        """Переключить агента в режим обучения"""
        if self.agent is not None:
            self.agent.actor.train()
            self.agent.critic.train()

    def eval_mode(self):
        """Переключить агента в режим инференса"""
        if self.agent is not None:
            self.agent.actor.eval()
            self.agent.critic.eval()

    @staticmethod
    def create_from_checkpoint(
        checkpoint_path: str,
        state_dim: int = 2,
        action_dim: int = 1,
        max_action: float = 5.0,
        hidden_dim: int = 64,
        device: str = None
    ):
        """
        Создать TD3Policy с загрузкой весов из checkpoint.

        Parameters:
        -----------
        checkpoint_path : str
            Путь к файлу с весами (.pth)
        state_dim : int
            Размерность состояния
        action_dim : int
            Размерность действия
        max_action : float
            Максимальное действие
        hidden_dim : int
            Размер скрытых слоев
        device : str, optional
            Устройство ('cpu' или 'cuda')

        Returns:
        --------
        policy : TD3Policy
            TD3 политика с загруженными весами
        """
        if not TD3_AVAILABLE:
            raise ImportError("TD3 module not available. Cannot create agent.")

        # Создаем TD3 агента
        agent = TD3(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            hidden_dim=hidden_dim,
            device=device
        )

        # Создаем политику
        policy = TD3Policy(agent=agent, action_dim=action_dim, device=device)

        # Загружаем веса
        policy.load_weights(checkpoint_path)

        return policy
