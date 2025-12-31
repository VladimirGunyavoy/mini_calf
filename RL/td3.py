import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):
    """Actor network for TD3"""

    def __init__(self, state_dim, action_dim, max_action, hidden_dim=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)


class Critic(nn.Module):
    """Twin Critic networks for TD3"""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


class ReplayBuffer:
    """Simple replay buffer for storing transitions"""

    def __init__(self, state_dim, action_dim, max_size=100000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim))
        self.actions = np.zeros((max_size, action_dim))
        self.next_states = np.zeros((max_size, state_dim))
        self.rewards = np.zeros((max_size, 1))
        self.dones = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.states[ind]),
            torch.FloatTensor(self.actions[ind]),
            torch.FloatTensor(self.next_states[ind]),
            torch.FloatTensor(self.rewards[ind]),
            torch.FloatTensor(self.dones[ind])
        )


class TD3:
    """
    Simplified Twin Delayed Deep Deterministic Policy Gradient (TD3)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        hidden_dim=64,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        lr=3e-4,
        device=None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"TD3 using device: {self.device}")

        # Actor and target
        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic and target
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.total_it = 0

    def select_action(self, state, noise=0.0):
        """Select action with optional exploration noise"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()

        if noise > 0:
            action = action + np.random.normal(0, noise, size=self.action_dim)
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def select_action_batch(self, states, noise=0.0):
        """
        Select actions for a batch of states (optimized for visualization)

        Parameters:
        -----------
        states : np.ndarray
            Batch of states, shape (batch_size, state_dim)
        noise : float
            Exploration noise std (applied to all actions)

        Returns:
        --------
        np.ndarray
            Batch of actions, shape (batch_size, action_dim)
        """
        states_tensor = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            actions = self.actor(states_tensor).cpu().data.numpy()

        if noise > 0:
            actions = actions + np.random.normal(0, noise, size=actions.shape)
            actions = np.clip(actions, -self.max_action, self.max_action)

        return actions

    def train(self, replay_buffer, batch_size=64):
        """Train the TD3 agent"""
        self.total_it += 1

        # Sample from replay buffer
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        # Move to device
        state = state.to(self.device)
        action = action.to(self.device)
        next_state = next_state.to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)

        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute target Q
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.discount * target_q

        # Get current Q estimates
        current_q1, current_q2 = self.critic(state, action)

        # Critic loss
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        actor_loss = None
        if self.total_it % self.policy_freq == 0:
            # Actor loss
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()

            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if actor_loss is not None else None
        }

    def save(self, filename):
        """Save model weights"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filename)

    def load(self, filename):
        """Load model weights"""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
