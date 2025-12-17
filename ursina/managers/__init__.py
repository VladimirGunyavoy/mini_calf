"""
Managers - все менеджеры системы
"""

from .color_manager import ColorManager
from .input_manager import InputManager
from .window_manager import WindowManager
from .zoom_manager import ZoomManager
from .object_manager import ObjectManager
from .ui_manager import UIManager
# from .visuals_update_manager import VisualsUpdateManager  # REMOVED in Phase 1.1
from .general_object_manager import GeneralObjectManager

__all__ = [
    'ColorManager',
    'InputManager',
    'WindowManager',
    'ZoomManager',
    'ObjectManager',
    'UIManager',
    # 'VisualsUpdateManager',  # REMOVED in Phase 1.1
    'GeneralObjectManager'
]
