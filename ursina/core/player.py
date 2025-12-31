"""
Класс Player - игрок с возможностью свободного полёта
"""

from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController


class Player(FirstPersonController):
    """Игрок с возможностью свободного полёта"""
    
    def __init__(self):
        # Стартовая позиция и углы
        self.initial_position = Vec3(2.15, -0.7, -1.75)
        self.initial_rotation_y = -38
        self.initial_camera_rotation_x = 26
        
        super().__init__(
            position=self.initial_position,
            speed=8,
            mouse_sensitivity=Vec2(40, 40)
        )
        
        # Отключаем гравитацию для свободного полёта
        self.gravity = 0
        
        # Скорость вертикального перемещения
        self.fly_speed = 8
        
        # Настройки камеры
        camera.fov = 90
        
        # Устанавливаем начальные углы
        self.rotation_y = self.initial_rotation_y
        self.camera_pivot.rotation_x = self.initial_camera_rotation_x
        
    def update(self):
        """Обновление каждый кадр"""
        super().update()
        
        # Свободное перемещение вверх/вниз
        if held_keys['space']:
            self.y += self.fly_speed * time.dt
            
        if held_keys['shift']:
            self.y -= self.fly_speed * time.dt
    
    def reset_position(self):
        """Возвращает игрока в начальную позицию и углы"""
        self.position = self.initial_position
        self.rotation_y = self.initial_rotation_y
        self.camera_pivot.rotation_x = self.initial_camera_rotation_x
        print(f"Player reset to initial position: {self.initial_position}")








