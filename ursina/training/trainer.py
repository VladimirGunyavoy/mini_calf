"""
CALF Trainer
Encapsulates training logic for CALF agent
"""

import numpy as np
from pathlib import Path


class CALFTrainer:
    """
    Manages CALF agent training loop and state.
    """

    def __init__(self, calf_agent, env, replay_buffer, nominal_policy, config):
        """
        Initialize trainer.

        Parameters:
        -----------
        calf_agent : CALFController
            CALF controller agent
        env : PointMassEnv
            Training environment
        replay_buffer : ReplayBuffer
            Replay buffer for experience storage
        nominal_policy : callable
            Nominal safe policy (PD controller)
        config : TrainingConfig
            Training configuration
        """
        self.calf_agent = calf_agent
        self.env = env
        self.replay_buffer = replay_buffer
        self.nominal_policy = nominal_policy
        self.config = config

        # Training state
        self.current_state = env.reset()
        self.episode = 0
        self.total_steps = 0
        self.episode_reward = 0.0
        self.episode_length = 0

        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0

        # Training metrics
        self.avg_critic_loss = 0.0
        self.avg_actor_loss = 0.0
        self.training_started = False

        # Control
        self.paused = False

    def should_train(self):
        """Check if training should start (enough exploration)."""
        return self.total_steps >= self.config.start_training_step

    def train_step(self):
        """
        Execute one training step.

        Returns:
        --------
        tuple
            (next_state, done) - next state and episode termination flag
        """
        # Select action
        if not self.should_train():
            # Initial exploration with nominal policy
            action = self.nominal_policy(self.current_state)
        else:
            # CALF action selection
            self.training_started = True
            action = self.calf_agent.select_action(
                self.current_state,
                exploration_noise=self.config.exploration_noise
            )

        # Step environment
        next_state, reward, done, info = self.env.step(action)

        # Scale reward
        reward = reward * self.config.reward_scale

        # Check early termination
        distance = np.linalg.norm(next_state)
        position = abs(next_state[0])

        if distance < self.config.goal_epsilon:
            done = True
        elif position > self.config.boundary_limit:
            done = True
        elif self.episode_length >= self.config.max_steps_per_episode:
            done = True

        # Store transition
        self.replay_buffer.add(
            self.current_state,
            action,
            next_state,
            reward,
            float(done)
        )

        # Train agent
        if self.should_train():
            train_info = self.calf_agent.train(
                self.replay_buffer,
                self.config.batch_size
            )
            self.avg_critic_loss = train_info['critic_loss']
            if train_info['actor_loss'] is not None:
                self.avg_actor_loss = train_info['actor_loss']

        # Update state
        self.current_state = next_state
        self.episode_reward += reward
        self.episode_length += 1
        self.total_steps += 1

        return next_state, done, info

    def handle_episode_end(self, info):
        """
        Handle end of episode - statistics, logging, checkpoints.

        Parameters:
        -----------
        info : dict
            Episode info (contains 'in_goal', etc.)
        """
        # Record episode statistics
        self.episode_rewards.append(self.episode_reward)
        self.episode_lengths.append(self.episode_length)

        if info['in_goal']:
            self.success_count += 1

        # Reset CALF certificate
        if self.training_started:
            self.calf_agent.reset_certificate()

        # Reset for new episode
        self.current_state = self.env.reset()
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode += 1

        # Evaluation logging
        if self.episode % self.config.eval_interval == 0:
            self._print_evaluation()

        # Save checkpoint
        if self.episode % (self.config.eval_interval * 5) == 0:
            self._save_checkpoint()

        return self.current_state

    def _print_evaluation(self):
        """Print evaluation statistics."""
        avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
        success_rate = self.success_count / max(1, self.episode) * 100

        print(f"\nEpisode {self.episode} / {self.config.num_episodes}")
        print(f"  Avg Reward (last 10): {avg_reward:.2f} (scaled x{self.config.reward_scale})")
        print(f"  Success Rate: {success_rate:.1f}%")

        # CALF statistics
        calf_stats = self.calf_agent.get_statistics()
        print(f"  CALF Stats:")
        print(f"    P_relax: {calf_stats['P_relax']:.10f}")
        print(f"    Certification rate: {calf_stats['certification_rate']:.3f}")
        print(f"    Intervention rate: {calf_stats['intervention_rate']:.3f}")
        print(f"    Relax rate: {calf_stats['relax_rate']:.3f}")

    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_path = Path("checkpoints") / f"calf_episode_{self.episode}.pth"
        checkpoint_path.parent.mkdir(exist_ok=True)
        self.calf_agent.save(str(checkpoint_path))
        print(f"  Checkpoint saved: {checkpoint_path}")

    def is_complete(self):
        """Check if training is complete."""
        return self.episode >= self.config.num_episodes

    def finalize(self):
        """
        Finalize training - print summary, save final model.
        """
        avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
        success_rate = self.success_count / max(1, self.episode) * 100

        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Total Episodes: {self.episode}")
        print(f"Success Rate: {success_rate:.1f}%")

        # Final CALF statistics
        calf_stats = self.calf_agent.get_statistics()
        print(f"\nFinal CALF Statistics:")
        print(f"  P_relax: {calf_stats['P_relax']:.10f}")
        print(f"  Certification rate: {calf_stats['certification_rate']:.3f}")
        print(f"  Intervention rate: {calf_stats['intervention_rate']:.3f}")
        print(f"  Relax rate: {calf_stats['relax_rate']:.3f}")

        # Save final model
        final_path = Path("trained_calf_final.pth")
        self.calf_agent.save(str(final_path))
        print(f"\nFinal model saved: {final_path}")

    def get_stats(self):
        """
        Get current training statistics.

        Returns:
        --------
        dict
            Training statistics
        """
        return {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
            'avg_reward': np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0,
            'success_rate': self.success_count / max(1, self.episode) * 100,
            'avg_critic_loss': self.avg_critic_loss,
            'avg_actor_loss': self.avg_actor_loss,
            'buffer_size': self.replay_buffer.size,
            'training_started': self.training_started,
            'paused': self.paused
        }
