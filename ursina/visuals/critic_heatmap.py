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
        self.triangles_cached = None  # Cache triangle topology (doesn't change)

        # Performance metrics
        self.update_count = 0
        self.total_update_time = 0.0
        self.avg_update_time = 0.0
        self.ema_update_time = 0.0  # Exponential moving average
        self.ema_alpha = 0.1  # Smoothing factor for EMA (lower = smoother)

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

    def _create_triangles_topology(self):
        """Create triangle topology (called once, then cached)"""
        if self.triangles_cached is not None:
            return self.triangles_cached

        triangles = []

        # Create triangles (2 per grid cell) - counter-clockwise for correct facing
        for i in range(self.grid_size - 1):
            for j in range(self.grid_size - 1):
                # Vertex indices: vertex at (i,j) has index i*grid_size + j
                v0 = i * self.grid_size + j
                v1 = i * self.grid_size + (j + 1)
                v2 = (i + 1) * self.grid_size + j
                v3 = (i + 1) * self.grid_size + (j + 1)

                # Triangle 1 (reversed winding order for top-facing)
                triangles.extend([v0, v1, v2])

                # Triangle 2 (reversed winding order for top-facing)
                triangles.extend([v2, v1, v3])

        self.triangles_cached = triangles
        return triangles

    def _compute_vertices_and_colors(self):
        """Compute vertices and colors from current Q-values (vectorized)"""
        # Square Q-values for normalization (same as height computation)
        q_squared = self.q_values ** 2

        # Normalize squared Q-values to [0, 1] for color
        if self.q_max > self.q_min:
            q_normalized = (q_squared - self.q_min) / (self.q_max - self.q_min)
            q_normalized = np.clip(q_normalized, 0.0, 1.0)
            # NO INVERSION: low at origin (pit), high far away
            # q_normalized stays as is
        else:
            q_normalized = np.zeros_like(self.q_values)

        # Vectorized height computation
        # Normalize raw Q-values: (q^2 - min) / (max - min)
        if self.q_max > self.q_min:
            q_squared_normalized = (q_squared - self.q_min) / (self.q_max - self.q_min)
        else:
            q_squared_normalized = np.zeros_like(q_squared)

        q_squared_normalized = np.clip(q_squared_normalized, 0.0, 1.0)

        # Compute heights for all points at once
        heights = q_squared_normalized * self.height_scale + self.surface_epsilon

        # Flatten grids for vectorized operations
        x_flat = self.x_grid.flatten()
        v_flat = self.v_grid.flatten()
        heights_flat = heights.flatten()

        # Create position array (N x 3): [x, height, v]
        positions = np.stack([x_flat, heights_flat, v_flat], axis=1)

        # Apply zoom transformation (vectorized)
        positions_transformed = positions * self.a_transformation + self.b_translation

        # Convert to Vec3 list
        vertices = [Vec3(pos[0], pos[1], pos[2]) for pos in positions_transformed]

        # Compute colors (vectorized)
        colors = self._q_to_color_vectorized(q_normalized)

        return vertices, colors

    def _create_surface_mesh(self):
        """Create mesh for 3D surface (initial creation only)"""
        vertices, colors = self._compute_vertices_and_colors()
        triangles = self._create_triangles_topology()

        # Create mesh
        mesh = Mesh(
            vertices=vertices,
            triangles=triangles,
            colors=colors,
            mode='triangle'
        )

        return mesh

    def _update_surface_mesh(self):
        """Update existing mesh with new vertices and colors (fast path)"""
        if self.surface_entity is None or self.surface_entity.model is None:
            # No existing mesh, create new one
            self.rebuild()
            return

        # Compute new vertices and colors
        vertices, colors = self._compute_vertices_and_colors()

        # Update the mesh in-place
        self.surface_entity.model.vertices = vertices
        self.surface_entity.model.colors = colors
        self.surface_entity.model.generate()  # Regenerate GPU buffers

    def _q_to_color_vectorized(self, q_normalized_array):
        """
        Convert normalized Q-values to colors (vectorized version)
        Enhanced low-range gradient with alternating light/dark shades:
        Light Violet -> Dark Violet -> Light Blue -> Dark Blue -> Cyan -> Green -> Yellow -> Orange -> Red

        Parameters:
        -----------
        q_normalized_array : np.ndarray
            Array of normalized Q-values [0, 1], shape (grid_size, grid_size)

        Returns:
        --------
        list
            List of Vec4 colors, length (grid_size * grid_size)
        """
        # Flatten for easier processing
        q_flat = q_normalized_array.flatten()
        n = len(q_flat)

        # Initialize RGB arrays
        r = np.zeros(n)
        g = np.zeros(n)
        b = np.zeros(n)

        # 10-color gradient with more detail in low range (0.0 - 0.4)

        # Light Violet (0.0 to 0.08) - RGB(200, 150, 255)
        mask1 = q_flat < 0.08
        t1 = q_flat[mask1] / 0.08
        r[mask1] = 0.78
        g[mask1] = 0.59
        b[mask1] = 1.0

        # Dark Violet (0.08 to 0.16) - RGB(100, 50, 180)
        mask2 = (q_flat >= 0.08) & (q_flat < 0.16)
        t2 = (q_flat[mask2] - 0.08) / 0.08
        r[mask2] = 0.39
        g[mask2] = 0.20
        b[mask2] = 0.71

        # Light Blue (0.16 to 0.24) - RGB(100, 150, 255)
        mask3 = (q_flat >= 0.16) & (q_flat < 0.24)
        t3 = (q_flat[mask3] - 0.16) / 0.08
        r[mask3] = 0.39
        g[mask3] = 0.59
        b[mask3] = 1.0

        # Dark Blue (0.24 to 0.32) - RGB(0, 0, 180)
        mask4 = (q_flat >= 0.24) & (q_flat < 0.32)
        t4 = (q_flat[mask4] - 0.24) / 0.08
        r[mask4] = 0.0
        g[mask4] = 0.0
        b[mask4] = 0.71

        # Cyan (0.32 to 0.45) - RGB(0, 200, 255)
        mask5 = (q_flat >= 0.32) & (q_flat < 0.45)
        t5 = (q_flat[mask5] - 0.32) / 0.13
        r[mask5] = 0.0
        g[mask5] = 0.78
        b[mask5] = 1.0

        # Green (0.45 to 0.6)
        mask6 = (q_flat >= 0.45) & (q_flat < 0.6)
        t6 = (q_flat[mask6] - 0.45) / 0.15
        r[mask6] = 0.0
        g[mask6] = 1.0
        b[mask6] = 1.0 - t6 * 1.0

        # Yellow (0.6 to 0.75)
        mask7 = (q_flat >= 0.6) & (q_flat < 0.75)
        t7 = (q_flat[mask7] - 0.6) / 0.15
        r[mask7] = t7 * 1.0
        g[mask7] = 1.0
        b[mask7] = 0.0

        # Orange (0.75 to 0.875)
        mask8 = (q_flat >= 0.75) & (q_flat < 0.875)
        t8 = (q_flat[mask8] - 0.75) / 0.125
        r[mask8] = 1.0
        g[mask8] = 1.0 - t8 * 0.5
        b[mask8] = 0.0

        # Red (0.875 to 1.0)
        mask9 = q_flat >= 0.875
        t9 = (q_flat[mask9] - 0.875) / 0.125
        r[mask9] = 1.0
        g[mask9] = 0.5 - t9 * 0.5
        b[mask9] = 0.0

        # Create Vec4 array with semi-transparency (alpha=0.6)
        colors = [Vec4(r[i], g[i], b[i], 0.6) for i in range(n)]

        return colors

    def update(self, step):
        """
        Update heatmap if needed

        Parameters:
        -----------
        step : int
            Current simulation step
        """
        import time

        self.step_counter = step

        # Update only at specified frequency
        if step % self.update_frequency != 0:
            return

        # Start timing
        start_time = time.perf_counter()

        # Compute new Q-values
        self.q_values = self._compute_q_values()

        # Update surface (fast path - no destroy/create)
        if self.surface_entity is None:
            # First time, create the surface
            self.rebuild()
        else:
            # Update existing mesh in-place
            self._update_surface_mesh()

        # End timing and update metrics
        end_time = time.perf_counter()
        update_time = end_time - start_time

        # Update counters
        self.update_count += 1
        self.total_update_time += update_time

        # Arithmetic mean
        self.avg_update_time = self.total_update_time / self.update_count

        # Exponential moving average (more recent updates have higher weight)
        if self.update_count == 1:
            self.ema_update_time = update_time
        else:
            self.ema_update_time = self.ema_alpha * update_time + (1 - self.ema_alpha) * self.ema_update_time

    def rebuild(self):
        """Rebuild the surface mesh (full recreate - used for zoom transform or initial creation)"""
        # Destroy old surface (mesh уничтожится автоматически)
        if self.surface_entity is not None:
            destroy(self.surface_entity)
            self.surface_entity = None

        # Create new surface
        mesh = self._create_surface_mesh()
        self.surface_entity = Entity(
            model=mesh,
            double_sided=False,  # Only visible from top (prevents z-fighting)
            unlit=True,  # Don't apply lighting (use vertex colors)
            alpha=0.6  # Semi-transparent surface
        )

    def clear(self):
        """Clear the surface"""
        if self.surface_entity is not None:
            destroy(self.surface_entity)
            self.surface_entity = None

    def get_q_range(self):
        """Get current Q-value range"""
        return self.q_min, self.q_max

    def get_performance_stats(self):
        """
        Get performance statistics for heatmap updates

        Returns:
        --------
        dict
            Dictionary with performance metrics:
            - update_count: number of updates
            - avg_time_ms: average update time in milliseconds
            - ema_time_ms: EMA update time in milliseconds
            - avg_fps: average FPS (1/avg_time)
            - ema_fps: EMA FPS (1/ema_time)
        """
        avg_fps = 1.0 / self.avg_update_time if self.avg_update_time > 0 else 0
        ema_fps = 1.0 / self.ema_update_time if self.ema_update_time > 0 else 0

        return {
            'update_count': self.update_count,
            'avg_time_ms': self.avg_update_time * 1000,
            'ema_time_ms': self.ema_update_time * 1000,
            'avg_fps': avg_fps,
            'ema_fps': ema_fps
        }
    
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
        
        # NO INVERSION: minimum at origin (0,0), maximum far away (pit shape)
        # normalized stays as is
        
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

    def get_q_value_for_states_batch(self, states, use_cached=True):
        """
        Get Q-value heights for a batch of states (optimized for multiple agents)

        Parameters:
        -----------
        states : np.ndarray
            Batch of state vectors, shape (batch_size, 2) where each is [x, v]
        use_cached : bool
            If True, interpolate from cached grid Q-values (synchronized with surface)
            If False, query critic directly (may be out of sync with surface)

        Returns:
        --------
        np.ndarray
            Heights for agent positioning, shape (batch_size,)
        """
        batch_size = len(states)
        heights = np.zeros(batch_size)

        if use_cached:
            # Vectorized interpolation from cached grid
            for i, state in enumerate(states):
                q_value = self._interpolate_q_from_grid(state)
                heights[i] = self._compute_height_from_q(q_value, surface_offset=0.0)
        else:
            # Batch query critic directly
            with torch.no_grad():
                states_tensor = torch.FloatTensor(states).to(self.td3_agent.device)
                actions_tensor = torch.zeros(batch_size, self.td3_agent.action_dim).to(self.td3_agent.device)

                q1, q2 = self.td3_agent.critic(states_tensor, actions_tensor)
                q_values = torch.min(q1, q2).cpu().numpy().flatten()

            # Compute heights for all states
            for i, q_value in enumerate(q_values):
                heights[i] = self._compute_height_from_q(q_value, surface_offset=0.0)

        return heights
    
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
