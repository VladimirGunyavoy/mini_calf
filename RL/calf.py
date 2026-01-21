import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Support both direct and package imports
try:
    from td3 import TD3, ReplayBuffer
except ImportError:
    from RL.td3 import TD3, ReplayBuffer


class CALFController:
    """
    Simplified CALF (Critic as Lyapunov Function) controller

    Основная идея:
    - Критик используется как функция Ляпунова: -q(s,a) должен убывать к цели
    - Если критик сертифицирует действие (Ляпунов-условие выполнено), используем актор
    - Если нет, с вероятностью (1 - P_relax) используем безопасную политику π₀
    - P_relax убывает экспоненциально: P_relax *= λ_relax
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        nominal_policy,
        goal_region_radius=0.1,
        nu_bar=0.01,
        kappa_low_coef=0.5,
        kappa_up_coef=2.0,
        lambda_relax=0.99,
        hidden_dim=64,
        lr=3e-4,
        device=None,
        **td3_kwargs
    ):
        """
        Parameters:
        -----------
        state_dim : int
            Размерность состояния
        action_dim : int
            Размерность действия
        max_action : float
            Максимальное значение действия
        nominal_policy : callable
            Номинальная безопасная политика π₀(s) -> action
        goal_region_radius : float
            Радиус целевой области G
        nu_bar : float
            Порог убывания функции Ляпунова (ν̄ > 0)
        kappa_low_coef : float
            Коэффициент для нижней K_∞ функции
        kappa_up_coef : float
            Коэффициент для верхней K_∞ функции
        lambda_relax : float
            Relaxation factor λ_relax ∈ [0,1)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.nominal_policy = nominal_policy
        self.goal_region_radius = goal_region_radius
        self.nu_bar = nu_bar
        self.kappa_low_coef = kappa_low_coef
        self.kappa_up_coef = kappa_up_coef
        self.lambda_relax = lambda_relax

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # TD3 agent
        self.td3 = TD3(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            hidden_dim=hidden_dim,
            lr=lr,
            device=self.device,
            **td3_kwargs
        )

        # Сертифицированная тройка (s†, a†, w†)
        self.s_cert = None
        self.a_cert = None
        self.q_cert = None

        # Relax probability
        self.P_relax = lambda_relax

        # Statistics
        self.total_steps = 0
        self.nominal_interventions = 0
        self.relax_events = 0

        # Certified Q tracking
        self.q_cert_history = []
        self.action_sources = []  # 'td3', 'nominal', 'relax'

    def kappa_low(self, state_norm):
        """Нижняя K_∞ функция: κ_low(r) = C_low * r²"""
        return self.kappa_low_coef * (state_norm ** 2)

    def kappa_up(self, state_norm):
        """Верхняя K_∞ функция: κ_up(r) = C_up * r²"""
        return self.kappa_up_coef * (state_norm ** 2)

    def distance_to_goal(self, state):
        """Расстояние до целевой области G (расстояние до нуля)"""
        return np.linalg.norm(state)

    def check_lyapunov_certificate(self, state, action):
        """
        Проверить Ляпунов-сертификат для текущей пары (s, a)

        Условия:
        1. q(s,a) - q†(s†,a†) >= ν̄  (убывание -q)
        2. κ_low(|s|) <= -q(s,a) <= κ_up(|s|)  (K_∞ ограничения)

        Returns:
        --------
        bool : True если сертификат пройден
        """
        # Вычислить q(s, a)
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action_tensor = torch.FloatTensor(action.reshape(1, -1)).to(self.device)

        with torch.no_grad():
            q_current, _ = self.td3.critic(state_tensor, action_tensor)
            q_current = q_current.item()

        # Если нет сертифицированной тройки, инициализировать
        if self.q_cert is None:
            return True

        # Проверка условия 1: убывание функции Ляпунова
        # q(s,a) - q†(s†,a†) >= ν̄
        lyapunov_decrease = q_current - self.q_cert >= self.nu_bar

        # Проверка условия 2: K_∞ ограничения
        state_norm = self.distance_to_goal(state)
        k_low = self.kappa_low(state_norm)
        k_up = self.kappa_up(state_norm)

        # κ_low(|s|) <= -q(s,a) <= κ_up(|s|)
        k_infinity_bounds = (k_low <= -q_current <= k_up)

        return lyapunov_decrease and k_infinity_bounds

    def update_certificate(self, state, action, q_value=None):
        """
        Обновить сертифицированную тройку (s†, a†, q†)

        Parameters:
        -----------
        state : np.ndarray
            State vector
        action : np.ndarray
            Action vector
        q_value : float, optional
            Pre-computed Q-value (to avoid redundant forward pass in batch mode)
        """
        if q_value is None:
            # Fallback: compute Q-value (для одиночных вызовов select_action)
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action_tensor = torch.FloatTensor(action.reshape(1, -1)).to(self.device)

            with torch.no_grad():
                q_value_tensor, _ = self.td3.critic(state_tensor, action_tensor)
                q_value = q_value_tensor.item()

        self.s_cert = state.copy()
        self.a_cert = action.copy()
        self.q_cert = q_value
        self.q_cert_history.append(q_value)

    def reset_certificate(self):
        """
        Сбросить сертифицированную тройку (вызывать при начале нового эпизода)
        """
        self.s_cert = None
        self.a_cert = None
        self.q_cert = None

    def select_action(self, state, exploration_noise=0.0):
        """
        Выбрать действие согласно алгоритму CALF

        Steps:
        1. Получить действие от актора π_t
        2. Проверить Ляпунов-сертификат
        3. Если сертификат прошел -> использовать действие актора
        4. Если нет:
           - С вероятностью P_relax -> расслабиться и использовать актор
           - С вероятностью (1 - P_relax) -> использовать π₀
        5. Обновить P_relax
        """
        self.total_steps += 1

        # 1. Действие от актора
        action_actor = self.td3.select_action(state, noise=exploration_noise)

        # 2. Проверить сертификат
        certified = self.check_lyapunov_certificate(state, action_actor)

        if certified:
            # Обновить сертифицированную тройку
            self.update_certificate(state, action_actor)
            action = action_actor
            self.action_sources.append('td3')
        else:
            # Сертификат не прошел
            # Сэмплировать q ~ U[0,1]
            q = np.random.uniform(0, 1)

            if q >= self.P_relax:
                # Использовать номинальную политику π₀
                action = self.nominal_policy(state)
                self.nominal_interventions += 1
                self.action_sources.append('nominal')
            else:
                # Расслабиться - использовать актор
                action = action_actor
                self.relax_events += 1
                self.action_sources.append('relax')

        # 3. Обновить P_relax
        self.P_relax *= self.lambda_relax

        return action

    def select_action_batch(self, states, exploration_noise=0.0, return_modes=False, update_state=False):
        """
        Batch version of select_action for efficient multi-agent processing
        
        Key optimizations:
        1. Batch actor inference (1 GPU call instead of N)
        2. Batch critic inference (1 GPU call instead of N)
        3. Cached Q-values for update_certificate (avoid N redundant GPU calls)
        
        Parameters:
        -----------
        states : np.ndarray
            Batch of states, shape (batch_size, state_dim)
        exploration_noise : float
            Exploration noise std
        return_modes : bool
            If True, return (actions, modes) where modes is list of action sources
        update_state : bool
            If True, update internal state (P_relax, counters, certificate)
            Set to False for visualization agents to avoid corrupting training state
            
        Returns:
        --------
        actions : np.ndarray
            Batch of actions, shape (batch_size, action_dim)
        modes : list[str] (optional)
            Action sources: 'td3' (certified), 'relax' (uncertified but relaxed), 'fallback' (nominal policy)
        """
        states = np.asarray(states)
        batch_size = len(states)
        
        # OPTIMIZATION 1: Batch actor inference (1 call instead of N)
        actions_actor = self.td3.select_action_batch(states, noise=exploration_noise)
        
        # OPTIMIZATION 2: Batch critic inference (1 call instead of N)
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions_actor).to(self.device)
        
        with torch.no_grad():
            q_values, _ = self.td3.critic(states_tensor, actions_tensor)
            q_values = q_values.cpu().numpy().flatten()
        
        # Vectorized certificate checking
        certified = np.ones(batch_size, dtype=bool)
        
        if self.q_cert is not None:
            # Condition 1: Lyapunov decrease (q(s,a) - q† >= ν̄)
            lyapunov_ok = (q_values - self.q_cert) >= self.nu_bar
            
            # Condition 2: K_infinity bounds
            state_norms = np.linalg.norm(states, axis=1)
            k_low = self.kappa_low(state_norms)
            k_up = self.kappa_up(state_norms)
            # κ_low(|s|) <= -q(s,a) <= κ_up(|s|)
            k_infinity_ok = (k_low <= -q_values) & (-q_values <= k_up)
            
            certified = lyapunov_ok & k_infinity_ok
        
        # Snapshot P_relax for consistent mode selection (don't modify during loop)
        current_P_relax = self.P_relax
        
        # Vectorized random sampling for relax decisions
        random_q = np.random.uniform(0, 1, size=batch_size)
        
        # Determine modes vectorized
        modes_array = np.where(certified, 'td3', 
                               np.where(random_q >= current_P_relax, 'fallback', 'relax'))
        
        # Build final actions array
        final_actions = np.empty_like(actions_actor)
        
        # Certified actions - use actor
        final_actions[certified] = actions_actor[certified]
        
        # Relax actions - use actor  
        relax_mask = (modes_array == 'relax')
        final_actions[relax_mask] = actions_actor[relax_mask]
        
        # Fallback actions - use nominal policy (must loop, but rare)
        fallback_mask = (modes_array == 'fallback')
        fallback_indices = np.where(fallback_mask)[0]
        for i in fallback_indices:
            final_actions[i] = self.nominal_policy(states[i])
        
        # Update internal state only if requested (not for visualization)
        if update_state:
            self.total_steps += batch_size
            self.nominal_interventions += np.sum(fallback_mask)
            self.relax_events += np.sum(relax_mask)
            # Update P_relax once for entire batch
            self.P_relax *= (self.lambda_relax ** batch_size)
            # Update certificate with first certified action
            certified_indices = np.where(certified)[0]
            if len(certified_indices) > 0:
                idx = certified_indices[0]
                self.update_certificate(states[idx], actions_actor[idx], q_value=q_values[idx])
        
        if return_modes:
            return final_actions, modes_array.tolist()
        return final_actions

    def train(self, replay_buffer, batch_size=64):
        """Обучить TD3 agent"""
        return self.td3.train(replay_buffer, batch_size)

    def reset_relax_probability(self):
        """Сбросить relax probability (опционально при достижении цели)"""
        self.P_relax = self.lambda_relax

    def get_statistics(self):
        """Получить статистику работы CALF"""
        total = max(1, self.total_steps)
        intervention_rate = self.nominal_interventions / total
        relax_rate = self.relax_events / total
        certification_rate = 1.0 - intervention_rate - relax_rate
        
        return {
            'total_steps': self.total_steps,
            'nominal_interventions': self.nominal_interventions,
            'relax_events': self.relax_events,
            'P_relax': self.P_relax,
            'intervention_rate': intervention_rate,
            'relax_rate': relax_rate,
            'certification_rate': certification_rate,
            'q_cert_history': self.q_cert_history.copy()
        }

    def get_q_cert_history(self):
        """Получить историю сертифицированных Q-значений"""
        return self.q_cert_history.copy()

    def get_action_sources(self):
        """Получить историю источников действий"""
        return self.action_sources.copy()

    def clear_q_cert_history(self):
        """Очистить историю сертифицированных Q-значений"""
        self.q_cert_history = []
        self.action_sources = []

    def save(self, filename):
        """Сохранить модель"""
        self.td3.save(filename)
        # Сохранить дополнительные параметры CALF
        calf_params = {
            's_cert': self.s_cert,
            'a_cert': self.a_cert,
            'q_cert': self.q_cert,
            'P_relax': self.P_relax,
            'total_steps': self.total_steps,
            'nominal_interventions': self.nominal_interventions,
            'relax_events': self.relax_events
        }
        np.savez(filename.replace('.pth', '_calf.npz'), **calf_params)

    def load(self, filename):
        """Загрузить модель"""
        self.td3.load(filename)
        # Загрузить дополнительные параметры CALF
        calf_params = np.load(filename.replace('.pth', '_calf.npz'), allow_pickle=True)
        self.s_cert = calf_params['s_cert']
        self.a_cert = calf_params['a_cert']
        self.q_cert = calf_params['q_cert'].item() if calf_params['q_cert'].shape == () else calf_params['q_cert']
        self.P_relax = calf_params['P_relax'].item()
        self.total_steps = calf_params['total_steps'].item()
        self.nominal_interventions = calf_params['nominal_interventions'].item()
        self.relax_events = calf_params['relax_events'].item()
