"""
Reward Plotter
Creates and updates reward graphs as PNG files
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


class RewardPlotter:
    """
    Creates and periodically updates reward plot as PNG file.
    """

    def __init__(
        self,
        output_dir: str = 'plots',
        filename: str = 'reward_plot.png',
        update_frequency: int = 10,
        window_size: int = 10,
        figsize: tuple = (12, 6)
    ):
        """
        Initialize reward plotter.

        Parameters:
        -----------
        output_dir : str
            Directory to save plots
        filename : str
            Output filename for plot
        update_frequency : int
            Update plot every N episodes
        window_size : int
            Window size for moving average
        figsize : tuple
            Figure size (width, height) in inches
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.update_frequency = update_frequency
        self.window_size = window_size
        self.figsize = figsize

        # Data storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_flags = []

        # Timestamps
        self.start_time = datetime.now()
        self.last_update_episode = 0

    def add_episode(self, reward: float, length: int, success: bool = False):
        """
        Add episode data.

        Parameters:
        -----------
        reward : float
            Total episode reward
        length : int
            Episode length (steps)
        success : bool
            Whether episode reached goal
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.success_flags.append(success)

    def should_update(self) -> bool:
        """Check if plot should be updated."""
        current_episode = len(self.episode_rewards)
        return (current_episode > 0 and 
                current_episode % self.update_frequency == 0 and
                current_episode != self.last_update_episode)

    def update_plot(self, force: bool = False):
        """
        Update and save plot.

        Parameters:
        -----------
        force : bool
            Force update even if not at update_frequency
        """
        if not force and not self.should_update():
            return

        if len(self.episode_rewards) == 0:
            return

        self.last_update_episode = len(self.episode_rewards)

        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, sharex=True)

        episodes = np.arange(1, len(self.episode_rewards) + 1)
        rewards = np.array(self.episode_rewards)
        lengths = np.array(self.episode_lengths)

        # === Subplot 1: Reward ===
        ax1 = axes[0]

        # Raw rewards (semi-transparent)
        ax1.plot(episodes, rewards, 'b-', alpha=0.3, linewidth=1, label='Episode Reward')

        # Moving average (adaptive window for early training)
        if len(rewards) >= 2:
            # Use smaller window at start, full window later
            effective_window = min(self.window_size, len(rewards))
            moving_avg = self._moving_average(rewards, effective_window)
            ma_episodes = episodes[effective_window - 1:]
            ax1.plot(ma_episodes, moving_avg, 'b-', linewidth=2, 
                    label=f'Moving Avg ({effective_window})')

        # Mark successful episodes
        if any(self.success_flags):
            success_episodes = episodes[np.array(self.success_flags)]
            success_rewards = rewards[np.array(self.success_flags)]
            ax1.scatter(success_episodes, success_rewards, c='green', s=20, 
                       marker='^', zorder=5, label='Success')

        ax1.set_ylabel('Reward')
        ax1.set_title(f'Training Progress - {len(self.episode_rewards)} Episodes')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = self._get_stats_text(rewards)
        ax1.text(0.98, 0.95, stats_text, transform=ax1.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # === Subplot 2: Episode Length ===
        ax2 = axes[1]

        ax2.plot(episodes, lengths, 'orange', alpha=0.3, linewidth=1, label='Episode Length')

        if len(lengths) >= 2:
            effective_window = min(self.window_size, len(lengths))
            ma_lengths = self._moving_average(lengths, effective_window)
            ma_episodes_len = episodes[effective_window - 1:]
            ax2.plot(ma_episodes_len, ma_lengths, 'orange', linewidth=2,
                    label=f'Moving Avg ({effective_window})')

        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)

        # Tight layout and save
        plt.tight_layout()

        output_path = self.output_dir / self.filename
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        print(f"[Plot] Updated: {output_path}")

    def _moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving average."""
        return np.convolve(data, np.ones(window) / window, mode='valid')

    def _get_stats_text(self, rewards: np.ndarray) -> str:
        """Generate statistics text for plot."""
        n = len(rewards)
        
        # Recent stats (last window_size episodes)
        recent = rewards[-self.window_size:] if n >= self.window_size else rewards
        
        # Success rate
        recent_success = self.success_flags[-self.window_size:] if n >= self.window_size else self.success_flags
        success_rate = sum(recent_success) / len(recent_success) * 100 if recent_success else 0

        # Time elapsed
        elapsed = datetime.now() - self.start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        lines = [
            f"Episodes: {n}",
            f"Recent Avg: {np.mean(recent):.2f}",
            f"Recent Max: {np.max(recent):.2f}",
            f"Success: {success_rate:.1f}%",
            f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}"
        ]

        if n >= self.window_size:
            # Trend (compare last window to previous window)
            if n >= 2 * self.window_size:
                prev_avg = np.mean(rewards[-2*self.window_size:-self.window_size])
                curr_avg = np.mean(recent)
                trend = curr_avg - prev_avg
                trend_str = f"+{trend:.2f}" if trend >= 0 else f"{trend:.2f}"
                lines.append(f"Trend: {trend_str}")

        return '\n'.join(lines)

    def save_final(self):
        """Save final plot with additional statistics."""
        self.update_plot(force=True)

        # Also save data to numpy file
        data_path = self.output_dir / 'reward_data.npz'
        np.savez(
            data_path,
            episode_rewards=np.array(self.episode_rewards),
            episode_lengths=np.array(self.episode_lengths),
            success_flags=np.array(self.success_flags)
        )
        print(f"[Plot] Data saved: {data_path}")
