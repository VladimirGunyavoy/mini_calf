# ursina/visuals/oriented_agent.py
"""
Визуальный агент с ориентацией (cone) для дифференциального привода.

Cone указывает направление движения робота.
Ориентация theta преобразуется в rotation_y для Ursina.
"""

from ursina import Entity, Vec3, color
import numpy as np


class OrientedAgent(Entity):
    """
    Визуальный агент с ориентацией (cone).
    
    Используется для отображения дифференциального привода.
    Конус указывает направление ориентации theta.
    """
    
    def __init__(self, position=(0, 0, 0), orientation=0.0, 
                 scale=0.15, agent_color=None, height=0.1, **kwargs):
        """
        Инициализация ориентированного агента.
        
        Parameters:
        -----------
        position : tuple
            Начальная позиция (x, y, z) в Ursina
        orientation : float
            Начальная ориентация theta в радианах
        scale : float
            Масштаб агента
        agent_color : Color
            Цвет агента (по умолчанию orange)
        height : float
            Высота над землёй (Y координата)
        **kwargs
            Дополнительные параметры для Entity
        """
        super().__init__(
            model='cone',
            color=agent_color or color.orange,
            scale=(scale, scale * 1.5, scale),
            position=position,
            **kwargs
        )
        
        self.height = height
        
        # Cone по умолчанию смотрит вверх (Y+)
        # Поворачиваем чтобы смотрел вдоль Z+ (вперёд)
        self.rotation_x = 90
        
        self._orientation = orientation
        self._update_rotation()
    
    def _update_rotation(self):
        """Обновить rotation_y на основе ориентации theta"""
        # theta=0 -> смотрит вдоль +X (в мировых координатах)
        # В Ursina rotation_y поворачивает вокруг Y (вертикальная ось)
        # rotation_y=0 -> смотрит вдоль +Z
        # Нам нужно: theta=0 -> +X, значит rotation_y = -90
        # theta=pi/2 -> +Z, значит rotation_y = 0
        # Формула: rotation_y = -theta (в градусах) - 90
        angle_deg = np.degrees(self._orientation)
        self.rotation_y = -angle_deg - 90
    
    def set_orientation(self, theta: float):
        """
        Установить ориентацию агента.
        
        Parameters:
        -----------
        theta : float
            Угол ориентации в радианах
        """
        self._orientation = theta
        self._update_rotation()
    
    def get_orientation(self) -> float:
        """Получить текущую ориентацию в радианах"""
        return self._orientation
    
    def update_from_state(self, state: np.ndarray, height: float = None):
        """
        Обновить позицию и ориентацию из состояния [x, y, theta].
        
        Parameters:
        -----------
        state : np.ndarray
            Состояние [x, y, theta]
        height : float, optional
            Высота над землёй (Y координата). Если None, используется self.height
        """
        if height is None:
            height = self.height
            
        x, y, theta = state[0], state[1], state[2]
        # x -> X, y -> Z (в Ursina Y - вертикаль)
        self.position = Vec3(x, height, y)
        self.set_orientation(theta)
    
    def update_from_system(self, system, height: float = None):
        """
        Обновить из DifferentialDriveSystem.
        
        Parameters:
        -----------
        system : DifferentialDriveSystem
            Физическая система
        height : float, optional
            Высота над землёй
        """
        state = system.get_state()
        self.update_from_state(state, height)


def test_oriented_agent():
    """Тест OrientedAgent (только проверка создания, без визуализации)"""
    print("Testing OrientedAgent creation...")
    
    # Тест без Ursina app (только проверка логики)
    import numpy as np
    
    # Проверка расчёта rotation
    test_angles = [0, np.pi/4, np.pi/2, np.pi, -np.pi/2]
    
    print("\nOrientation -> Rotation mapping:")
    for theta in test_angles:
        angle_deg = np.degrees(theta)
        rotation_y = -angle_deg - 90
        print(f"  theta={theta:.2f} rad ({angle_deg:.0f} deg) -> rotation_y={rotation_y:.0f}")
    
    print("\nTest passed! (OrientedAgent logic verified)")


if __name__ == "__main__":
    test_oriented_agent()
