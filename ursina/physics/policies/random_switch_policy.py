"""
Random Switch Policy - Случайное переключение между политиками
===============================================================

Обёртка, которая на каждом шаге случайно выбирает одну из заданных политик.
Полезно для демонстрации смешанного поведения.
"""

import numpy as np
from typing import List, Dict
from .base_policy import Policy


class RandomSwitchPolicy(Policy):
    """
    Политика, которая случайно переключается между несколькими политиками.
    
    На каждом вызове get_action() случайно выбирается одна из политик
    согласно заданным вероятностям.
    
    Пример:
        pd_policy = PDPolicy(kp=1.0, kd=0.5)
        td3_policy = TD3Policy(action_dim=2)
        
        # 70% PD, 30% TD3
        switch_policy = RandomSwitchPolicy(
            policies=[pd_policy, td3_policy],
            probabilities=[0.7, 0.3]
        )
    """
    
    def __init__(self, policies: List[Policy], probabilities: List[float] = None):
        """
        Инициализация политики со случайным переключением.
        
        Parameters:
        -----------
        policies : List[Policy]
            Список политик для случайного выбора
        probabilities : List[float], optional
            Вероятности выбора каждой политики (должны суммироваться в 1.0)
            Если None, используется равномерное распределение
        """
        if len(policies) == 0:
            raise ValueError("Необходима хотя бы одна политика")
        
        self.policies = policies
        
        # Устанавливаем вероятности
        if probabilities is None:
            # Равномерное распределение
            self.probabilities = np.ones(len(policies)) / len(policies)
        else:
            if len(probabilities) != len(policies):
                raise ValueError("Длина probabilities должна совпадать с длиной policies")
            
            probs = np.array(probabilities)
            if not np.isclose(probs.sum(), 1.0):
                print(f"⚠️ Вероятности не суммируются в 1.0 (сумма={probs.sum():.3f}), нормализуем")
                probs = probs / probs.sum()
            
            self.probabilities = probs
        
        # Статистика использования
        self.usage_counts = np.zeros(len(policies), dtype=int)
        self.total_calls = 0
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Случайно выбрать политику и получить действие.
        
        Parameters:
        -----------
        state : np.ndarray
            Состояние системы
        
        Returns:
        --------
        action : np.ndarray
            Действие от случайно выбранной политики
        """
        # Случайный выбор политики согласно вероятностям
        policy_idx = np.random.choice(len(self.policies), p=self.probabilities)
        
        # Обновляем статистику
        self.usage_counts[policy_idx] += 1
        self.total_calls += 1
        
        # Получаем действие от выбранной политики
        selected_policy = self.policies[policy_idx]
        action = selected_policy.get_action(state)
        
        return action
    
    def get_actions_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Получить действия для батча состояний.
        Каждое состояние обрабатывается независимо со случайным выбором политики.
        
        Parameters:
        -----------
        states : np.ndarray
            Батч состояний, shape (batch_size, state_dim)
        
        Returns:
        --------
        actions : np.ndarray
            Батч действий, shape (batch_size, action_dim)
        """
        # Для каждого состояния независимо выбираем политику
        return np.array([self.get_action(s) for s in states])
    
    def reset(self):
        """Сбросить статистику использования."""
        self.usage_counts = np.zeros(len(self.policies), dtype=int)
        self.total_calls = 0
        
        # Сбрасываем внутренние состояния политик
        for policy in self.policies:
            if hasattr(policy, 'reset'):
                policy.reset()
    
    def print_stats(self):
        """Вывести статистику использования политик."""
        print("\n--- RandomSwitchPolicy Statistics ---")
        print(f"Total calls: {self.total_calls}")
        for i, policy in enumerate(self.policies):
            policy_name = policy.__class__.__name__
            count = self.usage_counts[i]
            prob_target = self.probabilities[i] * 100
            prob_actual = (count / self.total_calls * 100) if self.total_calls > 0 else 0
            print(f"  {policy_name}: {count} calls ({prob_actual:.1f}%, target {prob_target:.1f}%)")
        print("-------------------------------------")
