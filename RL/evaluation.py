"""
Модуль для evaluation и визуализации прогресса обучения CALF
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
import os
from datetime import datetime
import shutil


class TrainingEvaluator:
    """
    Класс для evaluation и визуализации во время обучения
    """

    def __init__(self, save_dir='trainings', run_id=None):
        """
        Parameters:
        -----------
        save_dir : str
            Базовая папка для сохранения результатов
        run_id : str or None
            ID запуска (если None, создается автоматически)
        """
        self.save_dir = save_dir

        # Получить текущую дату и время
        now = datetime.now()

        # Создать структуру: год-месяц / неделя_NN / день_ГГГГ-ММ-ДД
        year_month = now.strftime('%Y-%m')  # 2025-12
        week_number = now.isocalendar()[1]  # номер недели в году
        week_dir = f"week_{week_number:02d}"
        day_dir = now.strftime('%Y-%m-%d')  # 2025-12-12

        # Базовый путь к папке дня
        day_path = os.path.join(save_dir, year_month, week_dir, day_dir)
        os.makedirs(day_path, exist_ok=True)

        # Создать ID запуска
        if run_id is None:
            # Найти следующий свободный номер в папке дня
            existing_runs = [d for d in os.listdir(day_path) if d.startswith('run_')]
            if existing_runs:
                run_numbers = [int(d.split('_')[1]) for d in existing_runs if d.split('_')[1].isdigit()]
                next_num = max(run_numbers) + 1 if run_numbers else 0
            else:
                next_num = 0
            self.run_id = f"run_{next_num:03d}"
        else:
            self.run_id = run_id

        # Создать структуру папок
        self.run_dir = os.path.join(day_path, self.run_id)
        self.last_eval_dir = os.path.join(self.run_dir, 'last_eval')
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.last_eval_dir, exist_ok=True)

        # Счетчик evaluations
        self.eval_count = 0

        # Буфер траекторий между evaluations
        self.trajectory_buffer = []

        # Буфер для certified Q history
        self.q_cert_buffer = []
        self.action_sources_buffer = []

        # Буфер для состояний replay buffer
        self.replay_states_buffer = []

        print(f"Training Evaluator initialized: {self.run_dir}")

    def add_trajectory(self, trajectory):
        """
        Добавить траекторию в буфер

        Parameters:
        -----------
        trajectory : np.array
            Траектория состояний [N x 2]
        """
        self.trajectory_buffer.append(trajectory.copy())

    def clear_trajectory_buffer(self):
        """Очистить буфер траекторий"""
        self.trajectory_buffer = []
        self.q_cert_buffer = []
        self.action_sources_buffer = []
        self.replay_states_buffer = []

    def add_q_cert_episode(self, q_cert_history):
        """Добавить историю certified Q для эпизода"""
        self.q_cert_buffer.append(q_cert_history)

    def add_action_sources_episode(self, action_sources):
        """Добавить историю источников действий для эпизода"""
        self.action_sources_buffer.append(action_sources)

    def add_replay_states(self, states):
        """Добавить состояния из replay buffer"""
        self.replay_states_buffer.extend(states)

    def plot_q_heatmap_with_trajectories(self, calf, env, trajectories,
                                          save_path, resolution=100):
        """
        Нарисовать тепловую карту Q-функции с CALF траекториями (цветные по источнику)
        """
        fig, ax = plt.subplots(figsize=(12, 12))

        # Создать сетку состояний
        x_range, v_range = (-3, 3), (-3, 3)
        x = np.linspace(x_range[0], x_range[1], resolution)
        v = np.linspace(v_range[0], v_range[1], resolution)
        X, V = np.meshgrid(x, v)

        # Батчевое вычисление Q-значений на GPU
        states_flat = np.column_stack([X.flatten(), V.flatten()])
        states_tensor = torch.FloatTensor(states_flat).to(calf.device)

        with torch.no_grad():
            actions_tensor = calf.td3.actor(states_tensor)
            q_values, _ = calf.td3.critic(states_tensor, actions_tensor)
            Q = q_values.cpu().numpy().reshape(resolution, resolution)

        # Нарисовать тепловую карту
        im = ax.contourf(X, V, Q, levels=20, cmap='viridis', alpha=0.8)
        plt.colorbar(im, ax=ax, label='Q-value', shrink=0.8)

        # Нарисовать траектории с цветом по источнику действия
        color_map = {'td3': 'green', 'nominal': 'red', 'relax': 'orange'}

        for i, traj in enumerate(trajectories):
            if i < len(self.action_sources_buffer):
                sources = self.action_sources_buffer[i]
                # Рисовать сегменты разными цветами
                for j in range(len(traj) - 1):
                    if j < len(sources):
                        color = color_map.get(sources[j], 'gray')
                        ax.plot(traj[j:j+2, 0], traj[j:j+2, 1],
                               color=color, linewidth=1, alpha=0.4)
            else:
                # Если нет источников, рисуем обычно
                ax.plot(traj[:, 0], traj[:, 1], 'gray', linewidth=0.5, alpha=0.3)

            ax.plot(traj[0, 0], traj[0, 1], 'ko', markersize=3, alpha=0.5)
            ax.plot(traj[-1, 0], traj[-1, 1], 'ko', markersize=3, alpha=0.5)

        # Добавить целевую область
        goal_circle = Circle((0, 0), env.goal_radius, color='red', fill=False,
                           linewidth=2, linestyle='--', label='Goal Region')
        ax.add_patch(goal_circle)

        # Легенда для источников действий
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', linewidth=2, label='TD3 (certified)'),
            Line2D([0], [0], color='red', linewidth=2, label='Nominal policy'),
            Line2D([0], [0], color='orange', linewidth=2, label='Relax'),
            Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Goal Region')
        ]
        ax.legend(handles=legend_elements, fontsize=10)

        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('Velocity', fontsize=12)
        ax.set_title(f'CALF Q-Function with {len(trajectories)} Trajectories', fontsize=14)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_q_cert_history(self, save_path):
        """
        Нарисовать историю сертифицированных Q-значений
        """
        if not self.q_cert_buffer:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        # Нарисовать каждый эпизод
        for i, q_hist in enumerate(self.q_cert_buffer):
            if len(q_hist) > 0:
                ax.plot(q_hist, alpha=0.5, linewidth=1)

        ax.set_xlabel('Step within episode', fontsize=12)
        ax.set_ylabel('Certified Q value', fontsize=12)
        ax.set_title(f'Certified Q History ({len(self.q_cert_buffer)} episodes)', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_replay_buffer_coverage(self, save_path, resolution=100):
        """
        Визуализировать покрытие пространства состояний replay buffer
        """
        if not self.replay_states_buffer:
            return

        fig, ax = plt.subplots(figsize=(12, 12))

        # Создать сетку для подсчета покрытия
        x_range, v_range = (-3, 3), (-3, 3)
        x_bins = np.linspace(x_range[0], x_range[1], resolution + 1)
        v_bins = np.linspace(v_range[0], v_range[1], resolution + 1)

        # Подсчитать сколько точек в каждой ячейке
        states = np.array(self.replay_states_buffer)
        coverage, x_edges, v_edges = np.histogram2d(
            states[:, 0], states[:, 1],
            bins=[x_bins, v_bins]
        )

        # Подсчитать процент покрытия
        total_cells = resolution * resolution
        covered_cells = np.sum(coverage > 0)
        coverage_percent = (covered_cells / total_cells) * 100

        # Нарисовать тепловую карту покрытия
        im = ax.imshow(coverage.T, origin='lower', extent=[x_range[0], x_range[1], v_range[0], v_range[1]],
                      cmap='plasma', aspect='auto', interpolation='nearest')
        plt.colorbar(im, ax=ax, label='Number of samples', shrink=0.8)

        # Нарисовать точки из буфера
        ax.scatter(states[:, 0], states[:, 1], c='cyan', s=1, alpha=0.1, label='Replay buffer states')

        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('Velocity', fontsize=12)
        ax.set_title(f'Replay Buffer Coverage\n{len(states)} states, {coverage_percent:.1f}% coverage ({covered_cells}/{total_cells} cells)',
                    fontsize=14)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        ax.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_td3_heatmap_with_trajectories(self, calf, env, save_path,
                                            num_test_episodes=20, max_steps=500, resolution=100):
        """
        TD3 траектории на тепловой карте Q-функции
        """
        fig, ax = plt.subplots(figsize=(12, 12))

        # Создать сетку состояний
        x_range, v_range = (-3, 3), (-3, 3)
        x = np.linspace(x_range[0], x_range[1], resolution)
        v = np.linspace(v_range[0], v_range[1], resolution)
        X, V = np.meshgrid(x, v)

        # Батчевое вычисление Q-значений на GPU
        states_flat = np.column_stack([X.flatten(), V.flatten()])
        states_tensor = torch.FloatTensor(states_flat).to(calf.device)

        with torch.no_grad():
            actions_tensor = calf.td3.actor(states_tensor)
            q_values, _ = calf.td3.critic(states_tensor, actions_tensor)
            Q = q_values.cpu().numpy().reshape(resolution, resolution)

        # Нарисовать тепловую карту
        im = ax.contourf(X, V, Q, levels=20, cmap='viridis', alpha=0.8)
        plt.colorbar(im, ax=ax, label='Q-value', shrink=0.8)

        # Генерировать TD3 траектории
        td3_trajectories = []
        for _ in range(num_test_episodes):
            state = env.reset()
            trajectory = [state.copy()]
            for step in range(max_steps):
                action = calf.td3.select_action(state, noise=0.0)
                state, _, done, _ = env.step(action)
                trajectory.append(state.copy())
                if done:
                    break
            td3_trajectories.append(np.array(trajectory))

        # Нарисовать траектории
        for traj in td3_trajectories:
            ax.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=0.8, alpha=0.5)
            ax.plot(traj[0, 0], traj[0, 1], 'wo', markersize=4, alpha=0.7, markeredgecolor='black', markeredgewidth=0.5)
            ax.plot(traj[-1, 0], traj[-1, 1], 'ko', markersize=4, alpha=0.7)

        # Добавить целевую область
        goal_circle = Circle((0, 0), env.goal_radius, color='red', fill=False,
                           linewidth=2, linestyle='--', label='Goal Region')
        ax.add_patch(goal_circle)

        # Легенда
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor='black', markersize=6, label='Start', linestyle=''),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                   markersize=6, label='Finish', linestyle=''),
            Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Goal Region')
        ]
        ax.legend(handles=legend_elements, fontsize=10)

        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('Velocity', fontsize=12)
        ax.set_title(f'TD3 Q-Function with {num_test_episodes} Trajectories', fontsize=14)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def test_policy(self, calf, env, use_calf=True, num_tests=5, max_steps=500):
        """
        Тестировать политику (CALF или чистый TD3)

        Returns:
        --------
        dict : статистика тестов
        """
        test_rewards = []
        test_lengths = []
        test_distances = []
        test_success = []

        for _ in range(num_tests):
            state = env.reset()
            episode_reward = 0
            steps = 0

            for step in range(max_steps):
                if use_calf:
                    action = calf.select_action(state, exploration_noise=0.0)
                else:
                    # Чистый TD3 без CALF
                    action = calf.td3.select_action(state, noise=0.0)

                next_state, reward, done, info = env.step(action)
                state = next_state
                episode_reward += reward
                steps += 1

                if done:
                    break

            test_rewards.append(episode_reward)
            test_lengths.append(steps)
            test_distances.append(info['distance_to_goal'])
            test_success.append(info['in_goal'])

        return {
            'avg_reward': np.mean(test_rewards),
            'std_reward': np.std(test_rewards),
            'avg_length': np.mean(test_lengths),
            'avg_distance': np.mean(test_distances),
            'success_rate': np.mean(test_success)
        }

    def evaluate(self, calf, env, episode, total_episodes,
                 episode_rewards, episode_lengths, final_distances,
                 intervention_rates, relax_rates, p_relax_history=None):
        """
        Выполнить evaluation и сохранить результаты

        Parameters:
        -----------
        calf : CALFController
        env : PointMassEnv
        episode : int
            Текущий эпизод
        total_episodes : int
            Всего эпизодов
        episode_rewards : list
        episode_lengths : list
        final_distances : list
        intervention_rates : list
        relax_rates : list
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION #{self.eval_count} at Episode {episode}/{total_episodes}")
        print(f"{'='*60}")

        # Тестировать обе политики
        print("Testing TD3 (without CALF)...")
        td3_stats = self.test_policy(calf, env, use_calf=False, num_tests=5)

        print("Testing CALF (with safety)...")
        calf_stats_test = self.test_policy(calf, env, use_calf=True, num_tests=5)

        print(f"  TD3:  Reward={td3_stats['avg_reward']:.2f}, Success={td3_stats['success_rate']:.2%}")
        print(f"  CALF: Reward={calf_stats_test['avg_reward']:.2f}, Success={calf_stats_test['success_rate']:.2%}")

        # Создать папку для этого evaluation
        eval_dir = os.path.join(self.run_dir, f'eval_{self.eval_count:03d}_ep{episode:04d}')
        os.makedirs(eval_dir, exist_ok=True)

        # 1. Сохранить графики обучения в корень run
        print(f"Plotting training progress...")
        self._plot_training_progress(
            episode_rewards, episode_lengths, final_distances,
            intervention_rates, relax_rates, p_relax_history,
            save_path=os.path.join(self.run_dir, 'training_progress.png')
        )

        # 2. Нарисовать Q-функцию с траекториями из буфера
        if self.trajectory_buffer:
            print(f"Plotting Q-function with {len(self.trajectory_buffer)} trajectories...")
            self.plot_q_heatmap_with_trajectories(
                calf, env, self.trajectory_buffer,
                save_path=os.path.join(eval_dir, 'q_function_trajectories.png')
            )

            print(f"Plotting TD3 heatmap with trajectories...")
            self.plot_td3_heatmap_with_trajectories(
                calf, env,
                save_path=os.path.join(eval_dir, 'td3_heatmap_trajectories.png')
            )

            print(f"Plotting certified Q history...")
            self.plot_q_cert_history(
                save_path=os.path.join(eval_dir, 'q_cert_history.png')
            )

            print(f"Plotting replay buffer coverage...")
            self.plot_replay_buffer_coverage(
                save_path=os.path.join(eval_dir, 'replay_buffer_coverage.png')
            )

        # 3. Сохранить веса модели
        print(f"Saving model weights...")
        model_path = os.path.join(eval_dir, 'model.pth')
        calf.save(model_path)

        # 4. Скопировать в last_eval (кроме training_progress)
        print(f"Copying to last_eval...")
        for filename in os.listdir(eval_dir):
            src = os.path.join(eval_dir, filename)
            dst = os.path.join(self.last_eval_dir, filename)
            shutil.copy2(src, dst)

        # Скопировать training_progress из корня в last_eval
        training_progress_src = os.path.join(self.run_dir, 'training_progress.png')
        training_progress_dst = os.path.join(self.last_eval_dir, 'training_progress.png')
        if os.path.exists(training_progress_src):
            shutil.copy2(training_progress_src, training_progress_dst)

        # 5. Сохранить статистику
        calf_stats = calf.get_statistics()
        stats_path = os.path.join(eval_dir, 'stats.txt')
        with open(stats_path, 'w') as f:
            f.write(f"Evaluation #{self.eval_count}\n")
            f.write(f"Episode: {episode}/{total_episodes}\n")
            f.write(f"\nCALF Statistics:\n")
            f.write(f"  Total Steps: {calf_stats['total_steps']}\n")
            f.write(f"  Nominal Interventions: {calf_stats['nominal_interventions']}\n")
            f.write(f"  Relax Events: {calf_stats['relax_events']}\n")
            f.write(f"  P_relax: {calf_stats['P_relax']:.10f}\n")
            f.write(f"  Intervention Rate: {calf_stats['intervention_rate']:.4f}\n")
            f.write(f"  Relax Rate: {calf_stats['relax_rate']:.4f}\n")
            f.write(f"\nTraining Statistics:\n")
            f.write(f"  Trajectories in buffer: {len(self.trajectory_buffer)}\n")
            if episode_rewards:
                f.write(f"  Last 20 Avg Reward: {np.mean(episode_rewards[-20:]):.2f}\n")
                f.write(f"  Last 20 Avg Length: {np.mean(episode_lengths[-20:]):.0f}\n")
                f.write(f"  Last 20 Avg Distance: {np.mean(final_distances[-20:]):.4f}\n")
            f.write(f"\nModel:\n")
            f.write(f"  Saved to: model.pth\n")
            f.write(f"  Also saved: model_calf.npz (CALF parameters)\n")

        print(f"Evaluation saved to: {eval_dir}")
        print(f"Last eval updated: {self.last_eval_dir}")
        print(f"{'='*60}\n")

        # Увеличить счетчик и очистить буфер траекторий
        self.eval_count += 1
        self.clear_trajectory_buffer()

    def _plot_training_progress(self, episode_rewards, episode_lengths,
                                 final_distances, intervention_rates, relax_rates,
                                 p_relax_history, save_path):
        """Визуализировать прогресс обучения"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        axes = axes.flatten()

        window = 20

        # 1. Episode Rewards
        axes[0].plot(episode_rewards, alpha=0.3, color='blue', label='Raw')
        if len(episode_rewards) >= window:
            moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            axes[0].plot(range(window-1, len(episode_rewards)), moving_avg,
                           color='red', linewidth=2, label=f'MA({window})')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('Episode Rewards')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Episode Lengths
        axes[1].plot(episode_lengths, alpha=0.3, color='green', label='Raw')
        if len(episode_lengths) >= window:
            moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
            axes[1].plot(range(window-1, len(episode_lengths)), moving_avg,
                           color='red', linewidth=2, label=f'MA({window})')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Steps')
        axes[1].set_title('Episode Lengths')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. Final Distances
        axes[2].plot(final_distances, alpha=0.3, color='purple', label='Raw')
        if len(final_distances) >= window:
            moving_avg = np.convolve(final_distances, np.ones(window)/window, mode='valid')
            axes[2].plot(range(window-1, len(final_distances)), moving_avg,
                           color='red', linewidth=2, label=f'MA({window})')
        axes[2].axhline(0.1, color='red', linestyle='--', alpha=0.5, label='Goal Radius')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Distance')
        axes[2].set_title('Final Distance to Goal')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale('log')

        # 4. Combined Rates
        axes[3].plot(intervention_rates, alpha=0.7, label='Intervention', color='orange')
        axes[3].plot(relax_rates, alpha=0.7, label='Relax', color='brown')
        axes[3].set_xlabel('Episode')
        axes[3].set_ylabel('Rate')
        axes[3].set_title('Intervention & Relax Rates')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        # 5. P_relax History
        if p_relax_history and len(p_relax_history) > 0:
            axes[4].plot(p_relax_history, alpha=0.7, color='magenta', linewidth=2)
            axes[4].set_xlabel('Episode')
            axes[4].set_ylabel('P_relax')
            axes[4].set_title('Relax Probability Over Time')
            axes[4].grid(True, alpha=0.3)
        else:
            axes[4].text(0.5, 0.5, 'No P_relax data', ha='center', va='center', transform=axes[4].transAxes)
            axes[4].set_title('Relax Probability')

        # 6. Placeholder for future use
        axes[5].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    print("Training Evaluator module")
    print("Use this module for evaluation during training")
