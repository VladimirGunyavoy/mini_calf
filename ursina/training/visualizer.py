"""
Training Visualizer
Manages visualization of training agents, heatmaps, and UI displays
"""

import numpy as np
import torch
from ursina import Text, Vec4

from visuals import LineTrail


class VisualAgent:
    """Visual agent with integrated trajectory (ring buffer)"""

    def __init__(self, object_manager, zoom_manager, agent_id, max_trail_length=200, decimation=1, point_color=None, trail_colors=None):
        """
        Initialize visual agent.

        Parameters:
        -----------
        object_manager : ObjectManager
            Object manager for creating visual entities
        zoom_manager : ZoomManager
            Zoom manager for transformations
        agent_id : int
            Agent identifier
        max_trail_length : int
            Maximum trail length (ring buffer size)
        decimation : int
            Trail decimation factor (1 = no decimation)
        point_color : Vec4, optional
            Agent sphere color (default: light blue)
        trail_colors : dict, optional
            Custom trail colors {'td3': Vec4, 'relax': Vec4, 'fallback': Vec4}
        """
        self.zoom_manager = zoom_manager

        # Agent sphere
        if point_color is None:
            point_color = Vec4(0.3, 0.7, 1.0, 1)  # Light blue (default)

        self.visual_point = object_manager.create_object(
            name=f'calf_point_{agent_id}',
            model='sphere',
            position=(0, 0, 0),
            scale=0.06,
            color_val=point_color
        )

        # Ring buffer trajectory (LineTrail)
        self.trail = LineTrail(
            max_points=max_trail_length,
            line_thickness=2,
            decimation=decimation,
            rebuild_freq=8,
            mode_colors=trail_colors
        )

        # Register trail in ZoomManager
        zoom_manager.register_object(self.trail, f'trail_{agent_id}')

        # Real position (for zoom)
        self.real_position = np.array([0, 0, 0], dtype=float)

    def update_position(self, position, mode='td3'):
        """
        Update agent position and add point to trajectory.

        Parameters:
        -----------
        position : tuple or np.ndarray
            Position (x, y, z)
        mode : str
            Mode for color coding ('td3', 'relax', 'fallback'/'nominal')
        """
        # Convert position to numpy array
        if isinstance(position, (tuple, list)):
            real_pos = np.array(position, dtype=float)
        else:
            real_pos = position

        # Save real position (for zoom)
        self.real_position = real_pos

        # Get current transformations
        a_trans = self.zoom_manager.a_transformation
        b_trans = self.zoom_manager.b_translation

        # Update sphere position
        self.visual_point.real_position = self.real_position
        self.visual_point.apply_transform(a_trans, b_trans)

        # Mode mapping: CALF uses 'nominal', visualization uses 'fallback'
        display_mode = mode
        if mode == 'nominal':
            display_mode = 'fallback'

        # Update sphere color based on mode
        if display_mode == 'td3':
            self.visual_point.color = Vec4(0.3, 0.7, 1.0, 1)  # Light blue
        elif display_mode == 'relax':
            self.visual_point.color = Vec4(0.2, 0.8, 0.3, 1)  # Green
        elif display_mode == 'fallback':
            self.visual_point.color = Vec4(0.9, 0.2, 0.2, 1)  # Red

        # Add point to trajectory with same transformations
        self.trail.add_point(
            self.real_position,
            mode=display_mode,
            a_transform=a_trans,
            b_translate=b_trans
        )

    def clear_trail(self):
        """Clear trajectory."""
        self.trail.clear()


class TrainingVisualizer:
    """
    Manages visualization of training: visual agents, heatmap, UI text.
    """

    def __init__(self, object_manager, zoom_manager, config, device):
        """
        Initialize visualizer.

        Parameters:
        -----------
        object_manager : ObjectManager
            Object manager for creating visual entities
        zoom_manager : ZoomManager
            Zoom manager for transformations
        config : VisualizationConfig
            Visualization configuration
        device : torch.device
            PyTorch device for computations
        """
        self.object_manager = object_manager
        self.zoom_manager = zoom_manager
        self.config = config
        self.device = device

        # Visual agents and environments
        self.visual_agents = []
        self.visual_envs = []
        self.visual_step_counters = []

        # Training agent (orange, with mode switching)
        self.training_agent = None

        # Heatmap and grid
        self.critic_heatmap = None
        self.grid_overlay = None
        self.heatmap_visible = True
        self.grid_visible = True
        self.goal_arrow = None

        # UI elements
        self.stats_text = None
        self.legend_text = None

        # Q-certificate timeline
        self.q_cert_timeline = None
        self.q_cert_step_counter = 0
        self.q_cert_min_value = 0.0
        self.q_cert_max_value = -100.0
        self.q_cert_graph_origin = None
        self.q_cert_max_display_steps = 0

    def setup_visual_agents(self, visual_envs):
        """
        Setup visual agents for given environments.

        Parameters:
        -----------
        visual_envs : list[PointMassEnv]
            Visual environments
        """
        self.visual_envs = visual_envs
        self.visual_step_counters = [0] * len(visual_envs)

        # Create visual agents
        for i in range(len(visual_envs)):
            agent = VisualAgent(
                object_manager=self.object_manager,
                zoom_manager=self.zoom_manager,
                agent_id=i,
                max_trail_length=self.config.trail_max_length,
                decimation=self.config.trail_decimation
            )
            self.visual_agents.append(agent)

        print(f"{len(self.visual_agents)} visual agents initialized")

    def setup_training_agent(self):
        """Setup training agent visualization (yellow sphere with yellow trail)."""
        # Yellow trail colors for training agent
        training_trail_colors = {
            'td3': Vec4(1.0, 0.9, 0.0, 1),      # Yellow (certified)
            'relax': Vec4(0.2, 0.8, 0.3, 1),    # Green
            'fallback': Vec4(0.9, 0.2, 0.2, 1)  # Red (nominal)
        }
        self.training_agent = VisualAgent(
            object_manager=self.object_manager,
            zoom_manager=self.zoom_manager,
            agent_id=999,  # Special ID for training agent
            max_trail_length=self.config.trail_max_length,
            decimation=1,
            point_color=Vec4(1.0, 0.9, 0.0, 1),  # Yellow
            trail_colors=training_trail_colors
        )
        print("Training agent visualization initialized")

    def setup_heatmap(self, critic_heatmap, grid_overlay=None):
        """
        Setup critic heatmap and optional grid overlay.

        Parameters:
        -----------
        critic_heatmap : CriticHeatmap
            Critic heatmap visualization
        grid_overlay : GridOverlay, optional
            Grid overlay visualization
        """
        self.critic_heatmap = critic_heatmap
        self.grid_overlay = grid_overlay

    def setup_ui(self, config):
        """
        Setup UI text elements.

        Parameters:
        -----------
        config : TrainingConfig
            Training configuration for display
        """
        # Stats text
        self.stats_text = Text(
            text='',
            position=(-0.85, 0.47),
            scale=0.7,
            origin=(-0.5, 0.5),
            background=True
        )

        # Legend text
        self.legend_text = Text(
            text='TRAIL COLORS:\n'
                 '<blue>Blue</blue> = TD3 (certified)\n'
                 '<green>Green</green> = Relax (uncertified, relaxed)\n'
                 '<orange>Orange</orange> = Fallback (nominal policy)\n'
                 '\n'
                 'AGENTS:\n'
                 '<color:rgb(255,128,0)>Orange point</color> = Training agent (CALF mode switching)\n'
                 '<blue>Blue points</blue> = Visual agents (pure TD3 policy)',
            position=(0.4, 0.45),
            scale=0.7,
            origin=(-0.5, 0.5),
            background=True
        )

    def setup_q_cert_timeline(self, q_cert_timeline, graph_origin, max_display_steps):
        """
        Setup Q-certificate timeline visualization.

        Parameters:
        -----------
        q_cert_timeline : LineTrail
            Timeline trail for Q-certificate
        graph_origin : np.ndarray
            Graph origin position
        max_display_steps : int
            Maximum steps to display
        """
        self.q_cert_timeline = q_cert_timeline
        self.q_cert_graph_origin = graph_origin
        self.q_cert_max_display_steps = max_display_steps

    def get_agent_height(self, state):
        """
        Get Y coordinate for agent based on critic Q-value.

        Parameters:
        -----------
        state : np.ndarray
            State vector

        Returns:
        --------
        float
            Y coordinate (Q-value height + 1*epsilon)
        """
        if self.critic_heatmap is not None:
            q_height = self.critic_heatmap.get_q_value_for_state(state, use_cached=True)
            return q_height + 1 * self.config.agent_height_epsilon
        else:
            return 0.1

    def get_agent_heights_batch(self, states):
        """
        Get Y coordinates for a batch of agents (optimized).

        Parameters:
        -----------
        states : np.ndarray
            Batch of state vectors, shape (batch_size, state_dim)

        Returns:
        --------
        np.ndarray
            Y coordinates, shape (batch_size,)
        """
        if self.critic_heatmap is not None:
            q_heights = self.critic_heatmap.get_q_value_for_states_batch(states, use_cached=True)
            return q_heights + 1 * self.config.agent_height_epsilon
        else:
            return np.full(len(states), 0.1)

    def update_training_agent(self, state, calf_agent):
        """
        Update training agent visualization.

        Parameters:
        -----------
        state : np.ndarray
            Current state
        calf_agent : CALFController
            CALF agent (for mode info)
        """
        if self.training_agent is None:
            return

        # Get height for training agent
        train_agent_height = self.get_agent_height(state)
        x, v = state[0], state[1]
        train_position = (x, train_agent_height, v)

        # Determine mode from last action source
        if len(calf_agent.action_sources) > 0:
            mode = calf_agent.action_sources[-1]
        else:
            mode = 'td3'

        self.training_agent.update_position(train_position, mode=mode)
        # Training agent: yellow for certified/relax, red for nominal intervention
        if mode == 'nominal':
            self.training_agent.visual_point.color = Vec4(0.9, 0.2, 0.2, 1)  # Red
        else:
            self.training_agent.visual_point.color = Vec4(1.0, 0.9, 0.0, 1)  # Yellow

    def update_q_cert_timeline(self, calf_agent):
        """
        Update Q-certificate timeline visualization.

        Parameters:
        -----------
        calf_agent : CALFController
            CALF agent (for Q-cert value and mode)
        """
        if self.q_cert_timeline is None or calf_agent.q_cert is None:
            return

        self.q_cert_step_counter += 1

        # Update min/max Q-cert values for scaling
        self.q_cert_min_value = min(self.q_cert_min_value, calf_agent.q_cert)
        self.q_cert_max_value = max(self.q_cert_max_value, calf_agent.q_cert)

        # Normalize X to [0, 2]
        graph_x = 2.0 * (self.q_cert_step_counter / self.q_cert_max_display_steps)

        # Normalize Y to [0, 2]
        if abs(self.q_cert_max_value - self.q_cert_min_value) > 0.001:
            graph_y = 2.0 * (calf_agent.q_cert - self.q_cert_min_value) / (self.q_cert_max_value - self.q_cert_min_value)
        else:
            graph_y = 1.0

        # Transform to world coordinates
        graph_position = self.q_cert_graph_origin + np.array([graph_x, graph_y, 0.0])

        # Get current mode
        if len(calf_agent.action_sources) > 0:
            mode = calf_agent.action_sources[-1]
        else:
            mode = 'td3'

        # Map mode for visualization
        display_mode = mode if mode != 'nominal' else 'fallback'

        # Add point to timeline
        a_trans = self.zoom_manager.a_transformation
        b_trans = self.zoom_manager.b_translation
        self.q_cert_timeline.add_point(
            graph_position,
            mode=display_mode,
            a_transform=a_trans,
            b_translate=b_trans
        )

    def clear_training_agent_trail(self):
        """Clear training agent trail."""
        if self.training_agent:
            self.training_agent.clear_trail()

    def clear_q_cert_timeline(self):
        """Clear Q-certificate timeline."""
        if self.q_cert_timeline:
            self.q_cert_timeline.clear()
        self.q_cert_step_counter = 0
        self.q_cert_min_value = 0.0
        self.q_cert_max_value = -100.0

    def update_visual_agents(self, policy, training_started, max_steps_per_episode, goal_epsilon, boundary_limit):
        """
        Update visual agents (batch processing).

        Parameters:
        -----------
        policy : TD3
            Policy for action selection (typically calf_agent.td3)
        training_started : bool
            Whether training has started
        max_steps_per_episode : int
            Maximum steps per episode
        goal_epsilon : float
            Goal distance threshold
        boundary_limit : float
            Boundary limit for early termination
        """
        if len(self.visual_envs) == 0:
            return

        # Collect all states
        vis_states = np.array([env.state for env in self.visual_envs])

        # Batch inference
        if training_started:
            vis_actions = policy.select_action_batch(vis_states, noise=0.0)
            vis_modes = ['td3'] * len(self.visual_envs)
        else:
            # Random actions during exploration
            vis_actions = np.random.uniform(
                -self.visual_envs[0].max_action,
                self.visual_envs[0].max_action,
                size=(len(self.visual_envs), self.visual_envs[0].action_dim)
            )
            vis_modes = ['td3'] * len(self.visual_envs)

        # Step all environments
        vis_next_states = []
        vis_done_flags = []

        for i in range(len(self.visual_envs)):
            vis_env = self.visual_envs[i]
            vis_action = vis_actions[i]

            # Increment step counter
            self.visual_step_counters[i] += 1

            # Step environment
            vis_next_state, vis_reward, vis_done, vis_info = vis_env.step(vis_action)

            # Check early termination
            vis_distance = np.linalg.norm(vis_next_state)
            vis_position = abs(vis_next_state[0])

            if vis_distance < goal_epsilon:
                vis_done = True
            elif vis_position > boundary_limit:
                vis_done = True
            elif self.visual_step_counters[i] >= max_steps_per_episode:
                vis_done = True

            vis_next_states.append(vis_next_state)
            vis_done_flags.append(vis_done)

        # Batch compute heights (OPTIMIZED: single batch call instead of N individual calls)
        vis_next_states_array = np.array(vis_next_states)
        vis_heights = self.get_agent_heights_batch(vis_next_states_array)

        # Get zoom transformations once (avoid multiple lookups)
        a_trans = self.zoom_manager.a_transformation
        b_trans = self.zoom_manager.b_translation

        # Update positions (still need loop for mode and done handling)
        for i in range(len(self.visual_envs)):
            vis_next_state = vis_next_states[i]
            vis_done = vis_done_flags[i]
            mode = vis_modes[i]

            x, v = vis_next_state[0], vis_next_state[1]
            y = vis_heights[i]
            position = (x, y, v)

            self.visual_agents[i].update_position(position, mode=mode)

            # Continuous flow: reset immediately when done
            if vis_done:
                self.visual_agents[i].clear_trail()
                self.visual_envs[i].reset()
                self.visual_step_counters[i] = 0

                # Update to new starting position
                new_state = self.visual_envs[i].state
                x, v = new_state[0], new_state[1]
                y = self.get_agent_height(new_state)
                new_position = (x, y, v)
                self.visual_agents[i].update_position(new_position, mode='td3')

    def update_heatmap(self, total_steps):
        """
        Update critic heatmap and grid overlay.

        Parameters:
        -----------
        total_steps : int
            Total training steps
        """
        if self.critic_heatmap is None or not self.heatmap_visible:
            return

        self.critic_heatmap.update(total_steps)

        # Update grid overlay (same frequency as heatmap)
        if self.grid_overlay is not None and self.grid_visible:
            if total_steps % self.config.heatmap_update_freq == 0:
                self.grid_overlay.update()

        # Update goal arrow
        if self.goal_arrow is None:
            self.goal_arrow = self.object_manager.create_object(
                name='goal_arrow',
                model='assets/arrow.obj',
                position=(0, 0, 0),
                rotation=(180, 0, 0),
                scale=0.5,
                color_val=Vec4(0.2, 0.8, 0.2, 1)
            )
            self.goal_arrow.unlit = True

        if self.goal_arrow is not None:
            goal_y = self.critic_heatmap.get_q_value_for_state(np.array([0, 0]), use_cached=True)
            self.goal_arrow.real_position = np.array([0, goal_y + 0.2, 0])  # Reduced offset from 0.5 to 0.2
            self.goal_arrow.apply_transform(
                self.zoom_manager.a_transformation,
                self.zoom_manager.b_translation
            )

    def update_stats_display(self, trainer, calf_agent, current_state, action, config):
        """
        Update statistics display.

        Parameters:
        -----------
        trainer : CALFTrainer
            Trainer for statistics
        calf_agent : CALFController
            CALF agent for CALF statistics
        current_state : np.ndarray
            Current state
        action : np.ndarray
            Current action
        config : TrainingConfig
            Training configuration
        """
        if self.stats_text is None:
            return

        stats = trainer.get_stats()
        calf_stats = calf_agent.get_statistics()
        session_stats = calf_agent.get_session_statistics()

        # Get Q-value range
        q_min, q_max = (0, 0)
        heatmap_perf = None
        if self.critic_heatmap is not None:
            q_min, q_max = self.critic_heatmap.get_q_range()
            heatmap_perf = self.critic_heatmap.get_performance_stats()

        # Get current Q-value
        current_q = 0.0
        q_cert_val = 0.0
        current_mode = 'exploration'
        k_low_val = 0.0
        k_up_val = 0.0
        neg_q_val = 0.0
        state_norm = 0.0

        if trainer.training_started:
            state_tensor = torch.FloatTensor(current_state.reshape(1, -1)).to(self.device)
            action_tensor = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
            with torch.no_grad():
                q_val, _ = calf_agent.td3.critic(state_tensor, action_tensor)
                current_q = q_val.item()

            if calf_agent.q_cert is not None:
                q_cert_val = calf_agent.q_cert

            if len(calf_agent.action_sources) > 0:
                current_mode = calf_agent.action_sources[-1]

            state_norm = np.linalg.norm(current_state)
            k_low_val = calf_agent.kappa_low(state_norm)
            k_up_val = calf_agent.kappa_up(state_norm)
            neg_q_val = -current_q

        # Build performance section
        perf_section = ""
        if heatmap_perf and heatmap_perf['update_count'] > 0:
            perf_section = f'''
=== Heatmap Performance ===
Updates: {heatmap_perf['update_count']}
Avg: {heatmap_perf['avg_time_ms']:.2f}ms ({heatmap_perf['avg_fps']:.1f} FPS)
EMA: {heatmap_perf['ema_time_ms']:.2f}ms ({heatmap_perf['ema_fps']:.1f} FPS)
'''

        self.stats_text.text = f'''CALF Training Progress

Episode: {stats['episode']} / {config.num_episodes}
Total Steps: {stats['total_steps']}
Visual Agents: {len(self.visual_agents)} (+/- to adjust)

=== Current Episode ===
Reward: {stats['episode_reward']:.2f} (x{config.reward_scale})
Length: {stats['episode_length']}

=== Overall ===
Avg Reward (10): {stats['avg_reward']:.2f} (x{config.reward_scale})
Success Rate: {stats['success_rate']:.1f}%

=== Training ===
Status: {"TRAINING" if stats['training_started'] else "EXPLORATION"}
Noise: {config.exploration_noise:.3f}
Buffer: {stats['buffer_size']}
Critic Loss: {stats['avg_critic_loss']:.4f}
Actor Loss: {stats['avg_actor_loss']:.4f}

=== CALF Statistics ===
Current Mode: {current_mode.upper()}
P_relax: {calf_stats['P_relax']:.8f}
Certification: {session_stats['certification_rate']:.3f}
Intervention: {session_stats['intervention_rate']:.3f}
Relax: {session_stats['relax_rate']:.3f}

=== Q-values ===
Q(s,a) current: {current_q:.4f}
Q_cert (Q†): {q_cert_val:.4f}
Delta_Q (Q - Q†): {current_q - q_cert_val:.4f}
Threshold nu_bar: {config.nu_bar:.4f}

=== K_infinity Bounds ===
|s| (state norm): {state_norm:.4f}
k_low: {k_low_val:.4f}
-Q(s,a): {neg_q_val:.4f}
k_up: {k_up_val:.4f}
Valid: {"YES" if k_low_val <= neg_q_val <= k_up_val else "NO"}

Grid Min: {q_min:.2f}
Grid Max: {q_max:.2f}{perf_section}
Press P to pause'''

    def add_visual_agent(self):
        """Add one visual agent dynamically."""
        from RL.simple_env import PointMassEnv

        print(f"\n[Agents] Adding 1 visual agent...")

        # Create new environment
        new_env = PointMassEnv(dt=0.01, max_action=5.0, goal_radius=0.1)
        new_env.reset()
        self.visual_envs.append(new_env)

        # Create new visual agent
        agent_id = len(self.visual_agents)
        new_agent = VisualAgent(
            object_manager=self.object_manager,
            zoom_manager=self.zoom_manager,
            agent_id=agent_id,
            max_trail_length=self.config.trail_max_length,
            decimation=self.config.trail_decimation
        )
        self.visual_agents.append(new_agent)
        self.visual_step_counters.append(0)

        print(f"[Agents] Total visual agents: {len(self.visual_agents)}")

    def remove_visual_agent(self):
        """Remove one visual agent dynamically."""
        if len(self.visual_agents) == 0:
            print(f"\n[Agents] No agents to remove!")
            return

        print(f"\n[Agents] Removing 1 visual agent...")

        # Remove last agent
        agent = self.visual_agents.pop()
        self.visual_envs.pop()
        self.visual_step_counters.pop()

        # Cleanup (destroy trail and sphere)
        agent.trail.clear()
        if hasattr(agent.visual_point, 'disable'):
            agent.visual_point.disable()

        print(f"[Agents] Total visual agents: {len(self.visual_agents)}")

    def toggle_heatmap_visibility(self):
        """Toggle heatmap visibility."""
        self.heatmap_visible = not self.heatmap_visible
        if self.critic_heatmap is not None:
            if hasattr(self.critic_heatmap, 'surface_entity') and self.critic_heatmap.surface_entity:
                self.critic_heatmap.surface_entity.visible = self.heatmap_visible
        print(f"\nHeatmap {'VISIBLE' if self.heatmap_visible else 'HIDDEN'}")

    def toggle_grid_visibility(self):
        """Toggle grid overlay visibility."""
        self.grid_visible = not self.grid_visible
        if self.grid_overlay is not None:
            for node in self.grid_overlay.node_entities:
                if node:
                    node.visible = self.grid_visible
            for line in self.grid_overlay.line_entities:
                if line:
                    line.visible = self.grid_visible
        print(f"\nGrid overlay {'VISIBLE' if self.grid_visible else 'HIDDEN'}")
