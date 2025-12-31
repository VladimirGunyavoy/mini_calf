"""
Grid overlay for critic heatmap
Shows a grid with nodes at integer coordinates and lines between them
"""

import numpy as np
import torch
from ursina import Entity, Mesh, Vec3, Vec4, destroy


class GridOverlay:
    """
    Grid overlay for critic heatmap surface
    
    - Grid nodes sampled from the same grid as heatmap
    - Lines connecting adjacent nodes
    - Nodes follow the heatmap height (Q-value)
    """
    
    def __init__(self, critic_heatmap, node_size=0.08, line_thickness=2, sample_step=3, grid_epsilon_multiplier=2.0):
        """
        Parameters:
        -----------
        critic_heatmap : CriticHeatmap
            Reference to heatmap for getting Q-values
        node_size : float
            Size of spheres at grid nodes
        line_thickness : float
            Thickness of grid lines
        sample_step : int
            Sample every N-th point from heatmap grid (e.g., 3 = every 3rd point)
        grid_epsilon_multiplier : float
            Grid height = surface_epsilon * multiplier (2.0 = 2*epsilon above ground)
        """
        self.critic_heatmap = critic_heatmap
        self.node_size = node_size
        self.line_thickness = line_thickness
        self.sample_step = sample_step
        self.grid_epsilon_multiplier = grid_epsilon_multiplier
        
        # Visual entities
        self.node_entities = []
        self.line_entities = []
        
        # Трансформации зума
        self.a_transformation = 1.0
        self.b_translation = np.array([0, 0, 0], dtype=float)
        self.zoom_manager = None
        
        # Create initial grid
        self._create_grid()
        
        print(f"[GridOverlay] Created grid overlay")
    
    def _create_grid(self):
        """Create grid nodes and lines using the SAME grid points as heatmap"""
        # Clear existing
        self._clear_entities()
        
        # Use EXACTLY the same grid as heatmap (but sample every N-th point)
        x_grid = self.critic_heatmap.x_grid
        v_grid = self.critic_heatmap.v_grid
        q_values = self.critic_heatmap.q_values
        grid_size = self.critic_heatmap.grid_size
        
        # Sample indices with step (e.g., 0, 3, 6, 9, ...)
        sample_indices = np.arange(0, grid_size, self.sample_step)
        
        # Store node positions for lines
        node_positions = {}
        
        # Create nodes at sampled grid points
        for i in sample_indices:
            for j in sample_indices:
                # Get exact coordinates from heatmap grid
                x = x_grid[i, j]
                v = v_grid[i, j]
                
                # Get Q-value from heatmap grid (SAME values as surface!)
                q_value = q_values[i, j]
                
                # Grid at same height as surface: both at epsilon
                grid_offset = self.critic_heatmap.surface_epsilon  # Same as surface
                height = self.critic_heatmap._compute_height_from_q(q_value, surface_offset=grid_offset)
                
                # Real position
                real_pos = np.array([x, height, v])
                
                # Apply zoom transformation
                transformed_pos = real_pos * self.a_transformation + self.b_translation
                position = Vec3(transformed_pos[0], transformed_pos[1], transformed_pos[2])
                
                # Create node sphere
                node = Entity(
                    model='sphere',
                    position=position,
                    scale=self.node_size,
                    color=Vec4(1, 1, 1, 1.0),  # White, fully opaque
                    unlit=True  # No lighting - always bright
                )
                self.node_entities.append(node)
                
                # Store for lines (using indices as key for proper ordering)
                node_positions[(i, j)] = position
        
        # Create lines (horizontal - along i axis, fixed j)
        for j in sample_indices:
            points = []
            for i in sample_indices:
                if (i, j) in node_positions:
                    points.append(node_positions[(i, j)])
            
            if len(points) >= 2:
                line = Entity(
                    model=Mesh(vertices=points, mode='line', thickness=self.line_thickness),
                    color=Vec4(1, 1, 1, 1.0),  # White, fully opaque
                    unlit=True  # No lighting - always bright
                )
                self.line_entities.append(line)
        
        # Create lines (vertical - along j axis, fixed i)
        for i in sample_indices:
            points = []
            for j in sample_indices:
                if (i, j) in node_positions:
                    points.append(node_positions[(i, j)])
            
            if len(points) >= 2:
                line = Entity(
                    model=Mesh(vertices=points, mode='line', thickness=self.line_thickness),
                    color=Vec4(1, 1, 1, 1.0),  # White, fully opaque
                    unlit=True  # No lighting - always bright
                )
                self.line_entities.append(line)
    
    def _clear_entities(self):
        """Clear all visual entities"""
        for node in self.node_entities:
            destroy(node)
        for line in self.line_entities:
            destroy(line)

        self.node_entities = []
        self.line_entities = []
    
    def update(self):
        """Update grid (rebuild with new Q-values)"""
        self._create_grid()
    
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
        # Перестроить с новой трансформацией
        self.update()
    
    def set_zoom_manager(self, zoom_manager):
        """Установить ссылку на ZoomManager"""
        self.zoom_manager = zoom_manager
    
    @property
    def enabled(self):
        """Для совместимости с ZoomManager"""
        return len(self.node_entities) > 0 or len(self.line_entities) > 0
    
    def clear(self):
        """Clear the grid"""
        self._clear_entities()
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            import sys
            if sys.meta_path is not None:
                self.clear()
        except:
            pass
