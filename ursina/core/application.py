"""
CALF Application Builder
Handles Ursina setup, scene initialization, and component management
"""

import numpy as np
import torch
from pathlib import Path
from ursina import Ursina, application, window, color

from core import Player, setup_scene
from managers import (
    InputManager, WindowManager, ZoomManager,
    ObjectManager, ColorManager, UIManager
)
from physics import VectorizedEnvironment
from RL.simple_env import PointMassEnv


class CALFApplication:
    """
    Application builder for CALF training visualization.
    Encapsulates Ursina setup and component initialization.
    """

    def __init__(self, config):
        """
        Initialize application with configuration.

        Parameters:
        -----------
        config : AppConfig
            Application configuration (training + visualization)
        """
        self.config = config
        self.app = None

        # Components
        self.player = None
        self.color_manager = None
        self.window_manager = None
        self.zoom_manager = None
        self.object_manager = None
        self.input_manager = None
        self.ui_manager = None

        # Scene elements
        self.ground = None
        self.grid = None
        self.lights = None
        self.frame = None

        # Device
        self.device = None

    def setup(self):
        """
        Setup Ursina application and all components.
        Returns self for chaining.
        """
        # Setup window BEFORE creating Ursina app
        WindowManager.setup_before_app(monitor="left")

        # Create Ursina app
        self.app = Ursina()

        # Initialize components
        self._setup_components()

        # Setup scene
        self._setup_scene()

        # Configure FPS counter
        self._setup_fps_counter()

        # Setup PyTorch device
        self._setup_device()

        # Print banner
        self._print_banner()

        return self

    def _setup_components(self):
        """Initialize all manager components."""
        self.player = Player()
        self.color_manager = ColorManager()
        self.window_manager = WindowManager(color_manager=self.color_manager, monitor="left")
        self.zoom_manager = ZoomManager(player=self.player)
        self.object_manager = ObjectManager(zoom_manager=self.zoom_manager)
        self.input_manager = InputManager(zoom_manager=self.zoom_manager, player=self.player)
        self.ui_manager = UIManager(
            color_manager=self.color_manager,
            player=self.player,
            zoom_manager=self.zoom_manager
        )

    def _setup_scene(self):
        """Setup 3D scene (ground, grid, lights, frame)."""
        self.ground, self.grid, self.lights, self.frame = setup_scene(
            self.color_manager,
            self.object_manager
        )

    def _setup_fps_counter(self):
        """Configure FPS counter display."""
        window.fps_counter.enabled = True
        window.fps_counter.position = (0.75, 0.48)
        window.fps_counter.color = color.white
        window.fps_counter.scale = 1.0

    def _setup_device(self):
        """Setup PyTorch device (CPU/CUDA)."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _print_banner(self):
        """Print application banner."""
        print("\n" + "="*70)
        print("CALF TRAINING WITH VISUALIZATION")
        print("Critic as Lyapunov Function - Mode Switching Visualization")
        print("="*70)
        print(f"\nDevice: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")

    def create_training_env(self):
        """
        Factory method to create training environment.

        Returns:
        --------
        PointMassEnv
            Training environment instance
        """
        env = PointMassEnv(dt=0.01, max_action=5.0, goal_radius=0.1)
        print(f"\nEnvironment: PointMassEnv")
        print(f"State dim: {env.state_dim}, Action dim: {env.action_dim}")
        print(f"Max action: {env.max_action}, Goal radius: {env.goal_radius}")
        return env

    def create_visual_envs(self, n_agents):
        """
        Factory method to create visual environments.

        Parameters:
        -----------
        n_agents : int
            Number of visual agents/environments

        Returns:
        --------
        list[PointMassEnv]
            List of visual environment instances
        """
        visual_envs = [PointMassEnv(dt=0.01, max_action=5.0, goal_radius=0.1)
                      for _ in range(n_agents)]

        # Reset all environments
        for ve in visual_envs:
            ve.reset()

        print(f"\n{n_agents} visual environments created")
        return visual_envs

    def update_managers(self):
        """Update all manager components (called every frame)."""
        if hasattr(self.input_manager, 'update'):
            self.input_manager.update()
        if hasattr(self.zoom_manager, 'update'):
            self.zoom_manager.update()
        if hasattr(self.object_manager, 'update'):
            self.object_manager.update()

    def handle_input(self, key):
        """
        Handle input keys (delegated from main input handler).

        Parameters:
        -----------
        key : str
            Key pressed
        """
        # Currently no application-level inputs
        # Can add debug keys, screenshot, etc. here
        pass

    def run(self):
        """Start the Ursina application loop."""
        self.app.run()

    def quit(self):
        """Quit the application."""
        application.quit()
