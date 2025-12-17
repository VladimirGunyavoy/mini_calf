"""
Модуль визуализации для CALF
- Тепловые карты Q-функции
- Траектории агента
- Графики обучения
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch


def plot_q_function_heatmap(calf, env, ax=None, resolution=100, x_range=(-3, 3), v_range=(-3, 3)):
    """
    Визуализировать Q-функцию как тепловую карту

    Parameters:
    -----------
    calf : CALFController
        Обученный CALF контроллер
    env : PointMassEnv
        Среда
    ax : matplotlib axis
        Ось для рисования (если None, создаст новую)
    resolution : int
        Разрешение сетки
    x_range : tuple
        Диапазон позиций
    v_range : tuple
        Диапазон скоростей
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Создать сетку состояний
    x = np.linspace(x_range[0], x_range[1], resolution)
    v = np.linspace(v_range[0], v_range[1], resolution)
    X, V = np.meshgrid(x, v)

    # Вычислить Q-значения для каждой точки сетки
    Q = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            state = np.array([X[i, j], V[i, j]])
            # Получить действие от актора
            action = calf.td3.select_action(state, noise=0.0)

            # Вычислить Q-значение
            state_tensor = torch.FloatTensor(state.reshape(1, -1))
            action_tensor = torch.FloatTensor(action.reshape(1, -1))

            with torch.no_grad():
                q_value, _ = calf.td3.critic(state_tensor, action_tensor)
                Q[i, j] = q_value.item()

    # Нарисовать тепловую карту
    im = ax.contourf(X, V, Q, levels=20, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Q-value')

    # Добавить целевую область
    goal_circle = Circle((0, 0), env.goal_radius, color='red', fill=False,
                         linewidth=2, linestyle='--', label='Goal Region')
    ax.add_patch(goal_circle)

    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title('Q-Function Heatmap')
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax


def plot_lyapunov_heatmap(calf, env, ax=None, resolution=100, x_range=(-3, 3), v_range=(-3, 3)):
    """
    Визуализировать функцию Ляпунова (-Q) как тепловую карту
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Создать сетку состояний
    x = np.linspace(x_range[0], x_range[1], resolution)
    v = np.linspace(v_range[0], v_range[1], resolution)
    X, V = np.meshgrid(x, v)

    # Вычислить -Q значения (функция Ляпунова)
    L = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            state = np.array([X[i, j], V[i, j]])
            action = calf.td3.select_action(state, noise=0.0)

            state_tensor = torch.FloatTensor(state.reshape(1, -1))
            action_tensor = torch.FloatTensor(action.reshape(1, -1))

            with torch.no_grad():
                q_value, _ = calf.td3.critic(state_tensor, action_tensor)
                L[i, j] = -q_value.item()  # Функция Ляпунова = -Q

    # Нарисовать тепловую карту
    im = ax.contourf(X, V, L, levels=20, cmap='RdYlGn_r')
    plt.colorbar(im, ax=ax, label='Lyapunov Function (-Q)')

    # Добавить целевую область
    goal_circle = Circle((0, 0), env.goal_radius, color='blue', fill=False,
                         linewidth=2, linestyle='--', label='Goal Region')
    ax.add_patch(goal_circle)

    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title('Lyapunov Function (-Q) Heatmap')
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax


def plot_trajectory_on_q_heatmap(calf, env, trajectory, ax=None, resolution=100,
                                  x_range=(-3, 3), v_range=(-3, 3)):
    """
    Нарисовать траекторию поверх тепловой карты Q-функции

    Parameters:
    -----------
    calf : CALFController
        Обученный CALF контроллер
    env : PointMassEnv
        Среда
    trajectory : np.array
        Траектория состояний [N x 2]
    """
    # Сначала нарисовать тепловую карту
    ax = plot_q_function_heatmap(calf, env, ax, resolution, x_range, v_range)

    # Добавить траекторию
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, alpha=0.8, label='Trajectory')
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End')

    ax.legend()
    ax.set_title('Trajectory on Q-Function Heatmap')

    return ax


def rollout_trajectory(calf, env, initial_state=None, max_steps=500, use_noise=False):
    """
    Собрать траекторию агента

    Returns:
    --------
    trajectory : np.array
        Траектория состояний
    actions : np.array
        Действия
    rewards : np.array
        Награды
    """
    state = env.reset(state=initial_state)
    trajectory = [state.copy()]
    actions = []
    rewards = []

    for step in range(max_steps):
        noise = 0.0 if not use_noise else 0.1
        action = calf.select_action(state, exploration_noise=noise)

        next_state, reward, done, info = env.step(action)

        trajectory.append(next_state.copy())
        actions.append(action)
        rewards.append(reward)

        state = next_state

        if done:
            break

    return np.array(trajectory), np.array(actions), np.array(rewards)


def plot_training_progress(episode_rewards, episode_lengths, final_distances,
                           intervention_rates, relax_rates, save_path='training_progress.png'):
    """
    Визуализировать прогресс обучения
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    window = 20

    # 1. Episode Rewards
    axes[0, 0].plot(episode_rewards, alpha=0.3, color='blue', label='Raw')
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(episode_rewards)), moving_avg,
                       color='red', linewidth=2, label=f'MA({window})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Episode Lengths
    axes[0, 1].plot(episode_lengths, alpha=0.3, color='green', label='Raw')
    if len(episode_lengths) >= window:
        moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(episode_lengths)), moving_avg,
                       color='red', linewidth=2, label=f'MA({window})')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Final Distances
    axes[0, 2].plot(final_distances, alpha=0.3, color='purple', label='Raw')
    if len(final_distances) >= window:
        moving_avg = np.convolve(final_distances, np.ones(window)/window, mode='valid')
        axes[0, 2].plot(range(window-1, len(final_distances)), moving_avg,
                       color='red', linewidth=2, label=f'MA({window})')
    axes[0, 2].axhline(0.1, color='red', linestyle='--', alpha=0.5, label='Goal Radius')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Distance')
    axes[0, 2].set_title('Final Distance to Goal')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_yscale('log')

    # 4. Intervention Rate
    axes[1, 0].plot(intervention_rates, alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Rate')
    axes[1, 0].set_title('Nominal Policy Intervention Rate')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Relax Rate
    axes[1, 1].plot(relax_rates, alpha=0.7, color='brown')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Rate')
    axes[1, 1].set_title('Relax Event Rate')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Combined Rates
    axes[1, 2].plot(intervention_rates, alpha=0.7, label='Intervention', color='orange')
    axes[1, 2].plot(relax_rates, alpha=0.7, label='Relax', color='brown')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Rate')
    axes[1, 2].set_title('Combined Rates')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training progress saved to {save_path}")

    return fig


def visualize_calf_results(calf, env, num_trajectories=5, save_dir='results'):
    """
    Полная визуализация результатов CALF

    Создает:
    1. Тепловую карту Q-функции с траекториями
    2. Тепловую карту функции Ляпунова с траекториями
    3. Фазовый портрет с несколькими траекториями
    4. График расстояния до цели во времени
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Собрать несколько траекторий из разных начальных точек
    initial_states = [
        [2.0, 1.0],
        [-2.0, -1.0],
        [1.5, -1.5],
        [-1.5, 1.5],
        [2.5, 0.5]
    ]

    trajectories = []
    for init_state in initial_states[:num_trajectories]:
        traj, _, _ = rollout_trajectory(calf, env, initial_state=init_state, max_steps=500)
        trajectories.append(traj)

    # 1. Q-функция с траекториями
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_q_function_heatmap(calf, env, ax=ax, resolution=100)

    for i, traj in enumerate(trajectories):
        ax.plot(traj[:, 0], traj[:, 1], linewidth=2, alpha=0.7, label=f'Traj {i+1}')
        ax.plot(traj[0, 0], traj[0, 1], 'o', markersize=8)
        ax.plot(traj[-1, 0], traj[-1, 1], 's', markersize=8)

    ax.legend(loc='upper right')
    ax.set_title('Q-Function with Trajectories')
    plt.savefig(f'{save_dir}/q_function_trajectories.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/q_function_trajectories.png")
    plt.close()

    # 2. Функция Ляпунова с траекториями
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_lyapunov_heatmap(calf, env, ax=ax, resolution=100)

    for i, traj in enumerate(trajectories):
        ax.plot(traj[:, 0], traj[:, 1], linewidth=2, alpha=0.7, label=f'Traj {i+1}')
        ax.plot(traj[0, 0], traj[0, 1], 'o', markersize=8)
        ax.plot(traj[-1, 0], traj[-1, 1], 's', markersize=8)

    ax.legend(loc='upper right')
    ax.set_title('Lyapunov Function (-Q) with Trajectories')
    plt.savefig(f'{save_dir}/lyapunov_trajectories.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/lyapunov_trajectories.png")
    plt.close()

    # 3. Фазовый портрет
    fig, ax = plt.subplots(figsize=(10, 10))

    for i, traj in enumerate(trajectories):
        ax.plot(traj[:, 0], traj[:, 1], linewidth=2, alpha=0.7, label=f'Traj {i+1}')
        ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10)
        ax.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=10)

    # Целевая область
    goal_circle = Circle((0, 0), env.goal_radius, color='red', fill=False,
                         linewidth=2, linestyle='--', label='Goal Region')
    ax.add_patch(goal_circle)

    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title('Phase Portrait')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')
    plt.savefig(f'{save_dir}/phase_portrait.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/phase_portrait.png")
    plt.close()

    # 4. Расстояние до цели во времени
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, traj in enumerate(trajectories):
        distances = np.linalg.norm(traj, axis=1)
        ax.plot(distances, linewidth=2, alpha=0.7, label=f'Traj {i+1}')

    ax.axhline(env.goal_radius, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Goal Radius')
    ax.set_xlabel('Step')
    ax.set_ylabel('Distance to Goal')
    ax.set_title('Distance to Goal over Time')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.legend()
    plt.savefig(f'{save_dir}/distance_over_time.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_dir}/distance_over_time.png")
    plt.close()

    print(f"\nAll visualizations saved to {save_dir}/")


if __name__ == "__main__":
    # Тест визуализации
    print("Модуль визуализации для CALF")
    print("Используйте функции из этого модуля для визуализации результатов обучения")
