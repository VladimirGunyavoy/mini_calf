"""
Пример обучения CALF (Critic as Lyapunov Function) с TD3
на простой задаче стабилизации точки
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from simple_env import PointMassEnv, pd_nominal_policy
from calf import CALFController
from td3 import ReplayBuffer
from visualize import plot_training_progress, visualize_calf_results
from evaluation import TrainingEvaluator


def train_calf(
    num_episodes=500,
    max_steps_per_episode=1000,
    batch_size=64,
    start_training_step=1000,
    exploration_noise=0.1,
    lambda_relax=0.99,
    eval_interval=20,
    seed=42
):
    """
    Обучить CALF контроллер

    Parameters:
    -----------
    num_episodes : int
        Количество эпизодов обучения
    max_steps_per_episode : int
        Максимальное количество шагов в эпизоде
    batch_size : int
        Размер батча для обучения
    start_training_step : int
        С какого шага начинать обучение
    exploration_noise : float
        Шум для исследования
    lambda_relax : float
        Relaxation factor
    eval_interval : int
        Интервал для evaluation (в эпизодах)
    seed : int
        Random seed
    """
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*60}\n")

    # Создать среду
    env = PointMassEnv(dt=0.01, max_action=5.0, goal_radius=0.1)

    # Номинальная безопасная политика (PD-контроллер)
    nominal_policy = pd_nominal_policy(max_action=env.max_action, kp=1.0, kd=1.0)

    # Создать CALF контроллер
    calf = CALFController(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        max_action=env.max_action,
        nominal_policy=nominal_policy,
        goal_region_radius=env.goal_radius,
        nu_bar=0.01,
        kappa_low_coef=0.5,
        kappa_up_coef=2.0,
        lambda_relax=lambda_relax,
        hidden_dim=64,
        lr=3e-4,
        device=device,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    )

    # Replay buffer
    replay_buffer = ReplayBuffer(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        max_size=100000
    )

    # Статистика обучения
    episode_rewards = []
    episode_lengths = []
    final_distances = []
    intervention_rates = []
    relax_rates = []
    p_relax_history = []

    total_steps = 0

    # Создать evaluator
    evaluator = TrainingEvaluator(save_dir='trainings')

    print("Начало обучения CALF...")
    print(f"Lambda_relax: {lambda_relax}")
    print(f"Номинальная политика: PD-контроллер")
    print(f"Evaluation interval: {eval_interval} episodes")
    print(f"Save directory: {evaluator.run_dir}")
    print("-" * 60)

    # Прогресс-бар
    pbar = tqdm(range(num_episodes), desc="Training")

    for episode in pbar:
        # Сбросить среду
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        critic_losses = []
        actor_losses = []

        # Сохранить траекторию для визуализации
        trajectory = [state.copy()]

        for step in range(max_steps_per_episode):
            # Выбрать действие
            if total_steps < start_training_step:
                # Начальное исследование с номинальной политикой
                action = nominal_policy(state)
            else:
                # CALF выбор действия
                action = calf.select_action(state, exploration_noise=exploration_noise)

            # Сделать шаг
            next_state, reward, done, info = env.step(action)

            # Сохранить в replay buffer
            replay_buffer.add(state, action, next_state, reward, float(done))

            # Обучение
            if total_steps >= start_training_step:
                train_info = calf.train(replay_buffer, batch_size)
                critic_losses.append(train_info['critic_loss'])
                if train_info['actor_loss'] is not None:
                    actor_losses.append(train_info['actor_loss'])

            # Обновить состояние
            state = next_state
            trajectory.append(state.copy())
            episode_reward += reward
            episode_length += 1
            total_steps += 1

            if done:
                break

        # Статистика эпизода
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        final_distances.append(info['distance_to_goal'])

        # Добавить траекторию в буфер evaluator
        evaluator.add_trajectory(np.array(trajectory))

        # Добавить certified Q history и action sources
        q_cert_history = calf.get_q_cert_history()
        action_sources = calf.get_action_sources()
        evaluator.add_q_cert_episode(q_cert_history)
        evaluator.add_action_sources_episode(action_sources)
        calf.clear_q_cert_history()

        # Статистика CALF
        calf_stats = calf.get_statistics()
        intervention_rates.append(calf_stats['intervention_rate'])
        relax_rates.append(calf_stats['relax_rate'])
        p_relax_history.append(calf_stats['P_relax'])

        # Обновить прогресс-бар
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0.0

        pbar.set_postfix({
            'R': f'{episode_reward:.1f}',
            'Len': episode_length,
            'Dist': f'{info["distance_to_goal"]:.4f}',
            'Goal': info['in_goal'],
            'P_relax': f'{calf_stats["P_relax"]:.6f}',
            'Interv': f'{calf_stats["intervention_rate"]:.3f}',
            'Loss': f'{avg_critic_loss:.4f}'
        })

        # Evaluation каждые eval_interval эпизодов
        if (episode + 1) % eval_interval == 0:
            # Добавить состояния из replay buffer
            if replay_buffer.size > 0:
                # Получить все состояния из буфера
                buffer_states = replay_buffer.states[:replay_buffer.ptr].copy()
                evaluator.add_replay_states(buffer_states)

            evaluator.evaluate(
                calf=calf,
                env=env,
                episode=episode + 1,
                total_episodes=num_episodes,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                final_distances=final_distances,
                intervention_rates=intervention_rates,
                relax_rates=relax_rates,
                p_relax_history=p_relax_history
            )

    # Финальная статистика
    print("\nОбучение завершено!")
    calf_stats = calf.get_statistics()
    print(f"Всего шагов: {calf_stats['total_steps']}")
    print(f"Вызовов номинальной политики: {calf_stats['nominal_interventions']}")
    print(f"Relax событий: {calf_stats['relax_events']}")
    print(f"Финальный P_relax: {calf_stats['P_relax']:.10f}")

    # Финальный evaluation (если не совпадает с интервалом)
    if num_episodes % eval_interval != 0:
        print("\nФинальный evaluation...")
        # Добавить состояния из replay buffer
        if replay_buffer.size > 0:
            buffer_states = replay_buffer.states[:replay_buffer.ptr].copy()
            evaluator.add_replay_states(buffer_states)

        evaluator.evaluate(
            calf=calf,
            env=env,
            episode=num_episodes,
            total_episodes=num_episodes,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            final_distances=final_distances,
            intervention_rates=intervention_rates,
            relax_rates=relax_rates,
            p_relax_history=p_relax_history
        )

    # Сохранить модель в директорию запуска
    import os
    model_path = os.path.join(evaluator.run_dir, 'calf_model.pth')
    calf.save(model_path)
    print(f"\nМодель сохранена в '{model_path}'")

    print(f"\nВсе результаты сохранены в: {evaluator.run_dir}")
    print(f"Последний evaluation: {evaluator.last_eval_dir}")

    return calf, env, episode_rewards, episode_lengths, evaluator


def test_trained_calf(calf, env, num_episodes=10):
    """Тестирование обученного CALF контроллера"""
    print("\nТестирование обученного контроллера...")
    print("-" * 60)

    trajectories = []

    for episode in range(num_episodes):
        state = env.reset()
        trajectory = [state.copy()]
        episode_reward = 0

        for step in range(1000):
            action = calf.select_action(state, exploration_noise=0.0)
            state, reward, done, info = env.step(action)
            trajectory.append(state.copy())
            episode_reward += reward

            if done:
                break

        trajectories.append(np.array(trajectory))

        print(f"Test Episode {episode + 1}:")
        print(f"  Steps: {len(trajectory)}")
        print(f"  Final Distance: {info['distance_to_goal']:.4f}")
        print(f"  In Goal: {info['in_goal']}")
        print(f"  Total Reward: {episode_reward:.2f}")

    # Визуализация траекторий
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    for traj in trajectories:
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.6)
    circle = plt.Circle((0, 0), env.goal_radius, color='r', alpha=0.2, label='Goal Region')
    plt.gca().add_patch(circle)
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Phase Portrait (Test)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()

    plt.subplot(1, 2, 2)
    for i, traj in enumerate(trajectories):
        distances = np.linalg.norm(traj, axis=1)
        plt.plot(distances, alpha=0.6, label=f'Episode {i+1}')
    plt.axhline(env.goal_radius, color='r', linestyle='--', alpha=0.5, label='Goal Radius')
    plt.xlabel('Step')
    plt.ylabel('Distance to Goal')
    plt.title('Distance over Time (Test)')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('calf_test_results.png', dpi=150)
    print("\nГрафик тестирования сохранен в 'calf_test_results.png'")
    plt.show()


if __name__ == "__main__":
    # Обучение
    calf, env, episode_rewards, episode_lengths, evaluator = train_calf(
        num_episodes=300,
        max_steps_per_episode=1000,
        batch_size=64,
        start_training_step=1000,
        exploration_noise=0.1,
        lambda_relax=0.99999,
        eval_interval=10,
        seed=42
    )

    print(f"\n{'='*60}")
    print(f"Обучение завершено!")
    print(f"Результаты сохранены в: {evaluator.run_dir}")
    print(f"Последние визуализации: {evaluator.last_eval_dir}")
    print(f"{'='*60}")
