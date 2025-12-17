"""
Critic Q-value heatmap visualization
Shows TD3 critic values as a colored 3D surface
"""

import numpy as np
from ursina import Entity, Mesh, Vec3, Vec4, destroy
import torch


class CriticHeatmap:
    """
    Visualize TD3 critic Q-values as a 3D surface

    - Grid: positions in (x, v) space
    - Height: normalized Q-value (0 to 1)
    - Color: Q-value (blue = low, red = high)
    """

    def __init__(self,
                 td3_agent,
                 grid_size=31,
                 x_range=(-5, 5),
                 v_range=(-5, 5),
                 height_scale=2.0,
                 update_frequency=100,
                 surface_epsilon=0.15):
        """
        Parameters:
        -----------
        td3_agent : TD3
            Trained TD3 agent with critic
        grid_size : int
            Number of grid points per dimension (31x31 = 961 points)
        x_range : tuple
            (min, max) for position axis
        v_range : tuple
            (min, max) for velocity axis
        height_scale : float
            Scale factor for height visualization
        update_frequency : int
            Update heatmap every N simulation steps
        surface_epsilon : float
            Height offset for surface above ground (epsilon)
        """
        self.td3_agent = td3_agent
        self.grid_size = grid_size
        self.x_range = x_range
        self.v_range = v_range
        self.height_scale = height_scale
        self.update_frequency = update_frequency
        self.surface_epsilon = surface_epsilon

        self.step_counter = 0
        self.surface_entity = None

        # Create grid
        self.x_grid, self.v_grid = self._create_grid()

        # Initial Q-values
        self.q_values = np.zeros((grid_size, grid_size))
        self.q_min = 0.0
        self.q_max = 1.0
        
        # Smoothing for min/max (exponential moving average)
        self.q_min_smooth = 0.0
        self.q_max_smooth = 1.0
        self.smooth_alpha = 0.1  # Smoothing factor (lower = smoother)
        
        # Трансформации зума
        self.a_transformation = 1.0
        self.b_translation = np.array([0, 0, 0], dtype=float)
        self.zoom_manager = None

        print(f"[CriticHeatmap] Created {grid_size}x{grid_size} grid")
        print(f"  X range: {x_range}")
        print(f"  V range: {v_range}")
        print(f"  Update frequency: {update_frequency} steps")

    def _create_grid(self):
        """Create 2D grid of (x, v) positions"""
        x = np.linspace(self.x_range[0], self.x_range[1], self.grid_size)
        v = np.linspace(self.v_range[0], self.v_range[1], self.grid_size)
        x_grid, v_grid = np.meshgrid(x, v)
        return x_grid, v_grid

    def _compute_q_values(self):
        """Compute Q-values for all grid points"""
        # Flatten grid
        states = np.stack([self.x_grid.flatten(), self.v_grid.flatten()], axis=1)

        # Get Q-values from critic (use zero action for state-value estimation)
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.td3_agent.device)
            actions_tensor = torch.zeros(len(states), self.td3_agent.action_dim).to(self.td3_agent.device)

            # Use Q1 from critic
            q1, q2 = self.td3_agent.critic(states_tensor, actions_tensor)
            q_values = torch.min(q1, q2).cpu().numpy().flatten()

        # Reshape to grid
        q_values = q_values.reshape(self.grid_size, self.grid_size)

        # Square Q-values first (before finding min/max)
        q_squared = q_values ** 2
        
        # Update min/max with exponential smoothing (on squared values)
        current_min = q_squared.min()
        current_max = q_squared.max()
        
        # Smooth update (reduces jumps)
        self.q_min_smooth = self.smooth_alpha * current_min + (1 - self.smooth_alpha) * self.q_min_smooth
        self.q_max_smooth = self.smooth_alpha * current_max + (1 - self.smooth_alpha) * self.q_max_smooth
        
        # Use smoothed values for normalization
        self.q_min = self.q_min_smooth
        self.q_max = self.q_max_smooth

        return q_values

    def _create_surface_mesh(self):
        """Create mesh for 3D surface"""
        vertices = []
        colors = []
        triangles = []

        # Square Q-values for normalization (same as height computation)
        q_squared = self.q_values ** 2
        
        # Normalize squared Q-values to [0, 1] for color
        if self.q_max > self.q_min:
            q_normalized = (q_squared - self.q_min) / (self.q_max - self.q_min)
            q_normalized = np.clip(q_normalized, 0.0, 1.0)
            # INVERT: to match inverted height (high at origin)
            q_normalized = 1.0 - q_normalized
        else:
            q_normalized = np.zeros_like(self.q_values)

        # Create vertices with height based on Q-value
        vertex_index = 0
        vertex_map = {}

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = self.x_grid[i, j]
                v = self.v_grid[i, j]
                
                # Get raw Q-value for this grid point
                q_raw = self.q_values[i, j]
                
                # Compute height with epsilon offset (surface above ground)
                height = self._compute_height_from_q(q_raw, surface_offset=self.surface_epsilon)
                
                # Get normalized value for color (from squared and normalized)
                q_norm_color = q_normalized[i, j]

                # Position in 3D space (применяем трансформацию зума)
                real_pos = np.array([x, height, v])
                transformed_pos = real_pos * self.a_transformation + self.b_translation
                position = Vec3(transformed_pos[0], transformed_pos[1], transformed_pos[2])
                vertices.append(position)

                # Color: blue (low Q^2) -> red (high Q^2)
                color = self._q_to_color(q_norm_color)
                colors.append(color)

                vertex_map[(i, j)] = vertex_index
                vertex_index += 1

        # Create triangles (2 per grid cell) - counter-clockwise for correct facing
        for i in range(self.grid_size - 1):
            for j in range(self.grid_size - 1):
                # Triangle 1 (reversed winding order for top-facing)
                v0 = vertex_map[(i, j)]
                v1 = vertex_map[(i, j + 1)]
                v2 = vertex_map[(i + 1, j)]
                triangles.extend([v0, v1, v2])

                # Triangle 2 (reversed winding order for top-facing)
                v0 = vertex_map[(i + 1, j)]
                v1 = vertex_map[(i, j + 1)]
                v2 = vertex_map[(i + 1, j + 1)]
                triangles.extend([v0, v1, v2])

        # Create mesh
        mesh = Mesh(
            vertices=vertices,
            triangles=triangles,
            colors=colors,
            mode='triangle'
        )

        return mesh

    def _q_to_color(self, q_normalized):
        """Convert normalized Q-value to color (blue -> cyan -> green -> yellow -> red)"""
        # Rainbow colormap: blue (0) -> red (1)
        if q_normalized < 0.25:
            # Blue -> Cyan
            t = q_normalized / 0.25
            r = 0.0
            g = t * 0.5
            b = 1.0
        elif q_normalized < 0.5:
            # Cyan -> Green
            t = (q_normalized - 0.25) / 0.25
            r = 0.0
            g = 0.5 + t * 0.5
            b = 1.0 - t
        elif q_normalized < 0.75:
            # Green -> Yellow
            t = (q_normalized - 0.5) / 0.25
            r = t
            g = 1.0
            b = 0.0
        else:
            # Yellow -> Red
            t = (q_normalized - 0.75) / 0.25
            r = 1.0
            g = 1.0 - t
            b = 0.0

        return Vec4(r, g, b, 1.0)  # Fully opaque

    def update(self, step):
        """
        Update heatmap if needed

        Parameters:
        -----------
        step : int
            Current simulation step
        """
        self.step_counter = step

        # Update only at specified frequency
        if step % self.update_frequency != 0:
            return

        # Compute new Q-values
        self.q_values = self._compute_q_values()

        # Rebuild surface
        self.rebuild()

    def rebuild(self):
        """Rebuild the surface mesh"""
        # Destroy old surface
        if self.surface_entity is not None:
            destroy(self.surface_entity)

        # Create new surface
        mesh = self._create_surface_mesh()
        self.surface_entity = Entity(
            model=mesh,
            double_sided=False,  # Only visible from top (prevents z-fighting)
            unlit=True,  # Don't apply lighting (use vertex colors)
            render_queue=0  # Render first to avoid flickering
        )

    def clear(self):
        """Clear the surface"""
        if self.surface_entity is not None:
            destroy(self.surface_entity)
            self.surface_entity = None

    def get_q_range(self):
        """Get current Q-value range"""
        return self.q_min, self.q_max
    
    def _compute_height_from_q(self, q_value, surface_offset=0.0):
        """
        Centralized function to compute height from Q-value
        
        Parameters:
        -----------
        q_value : float
            Raw Q-value from critic
        surface_offset : float
            Offset to apply (negative = lower, positive = higher)
        
        Returns:
        --------
        float
            Height for visualization
        """
        # Square Q-value
        q_squared = q_value ** 2
        
        # Normalize: (q^2 - min) / (max - min)
        if self.q_max > self.q_min:
            normalized = (q_squared - self.q_min) / (self.q_max - self.q_min)
        else:
            normalized = 0.0
        
        # Clamp to [0, 1]
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # INVERT: maximum at origin (0,0), minimum far away
        normalized = 1.0 - normalized
        
        # Scale and apply offset
        height = normalized * self.height_scale + surface_offset
        
        return height
    
    def get_q_value_for_state(self, state, use_cached=True):
        """
        Get Q-value height for a specific state (for agents/goal positioning)
        
        Parameters:
        -----------
        state : np.ndarray
            State vector [x, v]
        use_cached : bool
            If True, interpolate from cached grid Q-values (synchronized with surface)
            If False, query critic directly (may be out of sync with surface)
        
        Returns:
        --------
        float
            Height for agent positioning (no offset - agents at true Q-height)
        """
        if use_cached:
            # Interpolate from cached Q-values grid (synchronized with surface/grid)
            q_value = self._interpolate_q_from_grid(state)
        else:
            # Query critic directly (may be out of sync with surface)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.td3_agent.device)
                action_tensor = torch.zeros(1, self.td3_agent.action_dim).to(self.td3_agent.device)
                
                q1, q2 = self.td3_agent.critic(state_tensor, action_tensor)
                q_value = torch.min(q1, q2).cpu().numpy()[0, 0]
        
        # Use centralized height computation (no offset for agents)
        height = self._compute_height_from_q(q_value, surface_offset=0.0)
        
        return height
    
    def _interpolate_q_from_grid(self, state):
        """
        Interpolate Q-value from cached grid using bilinear interpolation
        
        Parameters:
        -----------
        state : np.ndarray
            State vector [x, v]
        
        Returns:
        --------
        float
            Interpolated Q-value
        
        Note:
        -----
        Grid indexing: q_values[i, j] where i=row (v-axis), j=col (x-axis)
        meshgrid creates: x_grid[i,j]=x[j], v_grid[i,j]=v[i]
        """
        x, v = state[0], state[1]
        
        # Clamp to grid bounds
        x = np.clip(x, self.x_range[0], self.x_range[1])
        v = np.clip(v, self.v_range[0], self.v_range[1])
        
        # Convert to grid coordinates (continuous)
        x_norm = (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
        v_norm = (v - self.v_range[0]) / (self.v_range[1] - self.v_range[0])
        
        # Grid indices (floating point)
        # col_float corresponds to x (2nd dimension, j)
        # row_float corresponds to v (1st dimension, i)
        col_float = x_norm * (self.grid_size - 1)
        row_float = v_norm * (self.grid_size - 1)
        
        # Get surrounding grid points
        col0 = int(np.floor(col_float))
        col1 = min(col0 + 1, self.grid_size - 1)
        row0 = int(np.floor(row_float))
        row1 = min(row0 + 1, self.grid_size - 1)
        
        # Interpolation weights
        wx = col_float - col0
        wv = row_float - row0
        
        # Bilinear interpolation: q_values[row, col]
        q00 = self.q_values[row0, col0]  # (v0, x0)
        q01 = self.q_values[row0, col1]  # (v0, x1)
        q10 = self.q_values[row1, col0]  # (v1, x0)
        q11 = self.q_values[row1, col1]  # (v1, x1)
        
        q_interp = (1 - wx) * (1 - wv) * q00 + \
                   wx * (1 - wv) * q01 + \
                   (1 - wx) * wv * q10 + \
                   wx * wv * q11
        
        return q_interp
    
    def apply_transform(self, a, b, **kwargs):
        """
        Применить трансформацию зума (для совместимости с ZoomManager)
        
        Parameters:
        -----------
        a : float
            Масштаб
        b : np.ndarray
            Смещение [x, y, z]
        """
        self.a_transformation = a
        self.b_translation = b
        # Перестроить с новой трансформацией (только если surface уже создана)
        if self.surface_entity is not None:
            self.rebuild()
    
    def set_zoom_manager(self, zoom_manager):
        """Установить ссылку на ZoomManager"""
        self.zoom_manager = zoom_manager
    
    @property
    def enabled(self):
        """Для совместимости с ZoomManager"""
        return self.surface_entity is not None
