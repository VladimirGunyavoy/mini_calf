"""
Менеджер управления вводом
Обрабатывает нажатия клавиш и действия пользователя
Архитектура: список словарей с кнопками, описаниями и функциями
"""

from ursina import *


class InputManager:
    def __init__(self, zoom_manager=None, player=None):
        """
        Args:
            zoom_manager: Опциональная ссылка на ZoomManager для привязки клавиш масштабирования
            player: Опциональная ссылка на Player для управления движением при freeze
        """
        # Список привязок клавиш
        # Формат: {'keys': str или tuple, 'description': str, 'action': callable}
        self.key_bindings = []

        # Ссылки на менеджеры
        self.zoom_manager = zoom_manager
        self.player = player

        # Состояние курсора
        self.cursor_locked = True

        # Регистрируем базовые привязки
        self._register_default_bindings()

        # Регистрируем привязки для zoom_manager, если он передан
        if self.zoom_manager:
            self._register_zoom_bindings()

    def _register_default_bindings(self):
        """Register default key bindings"""

        # Exit application (only Q)
        self.register_binding(
            keys='q',
            description="Exit application",
            action=self._quit_app
        )

        # Fullscreen mode
        self.register_binding(
            keys='f',
            description="Toggle fullscreen",
            action=self._toggle_fullscreen
        )

        # Mouse cursor
        self.register_binding(
            keys='m',
            description="Show/hide cursor",
            action=self._toggle_mouse_lock
        )

        # Debug info
        self.register_binding(
            keys='p',
            description="Print camera position",
            action=self._print_camera_position
        )

        # Freeze cursor toggle
        self.register_binding(
            keys='alt',
            description="Toggle freeze cursor",
            action=self._toggle_freeze
        )

    def _register_zoom_bindings(self):
        """Register ZoomManager bindings"""
        if not self.zoom_manager:
            return

        # Zoom with mouse wheel
        self.register_binding(
            keys='scroll up',
            description="Zoom in",
            action=self.zoom_manager.zoom_in
        )

        self.register_binding(
            keys='scroll down',
            description="Zoom out",
            action=self.zoom_manager.zoom_out
        )

        # Reset zoom and player position
        self.register_binding(
            keys='r',
            description="Reset zoom and player position",
            action=self._reset_all
        )

        # Object scale
        self.register_binding(
            keys='1',
            description="Increase objects scale",
            action=self.zoom_manager.increase_objects_scale
        )

        self.register_binding(
            keys='2',
            description="Decrease objects scale",
            action=self.zoom_manager.decrease_objects_scale
        )

    def set_zoom_manager(self, zoom_manager):
        """
        Устанавливает ZoomManager и регистрирует его привязки

        Args:
            zoom_manager: Экземпляр ZoomManager
        """
        self.zoom_manager = zoom_manager
        self._register_zoom_bindings()

    def register_binding(self, keys, description, action):
        """
        Регистрирует привязку клавиши к действию

        Args:
            keys: str или tuple - одна клавиша или несколько (для комбинаций)
            description: str - описание действия
            action: callable - функция, которая будет вызвана
        """
        self.key_bindings.append({
            'keys': keys if isinstance(keys, tuple) else (keys,),
            'description': description,
            'action': action
        })

    def handle_input(self, key):
        """
        Обработка нажатий клавиш
        Сначала проверяет комбинации (пары), потом индивидуальные клавиши
        """

        # Сортируем привязки: сначала комбинации (больше клавиш), потом одиночные
        sorted_bindings = sorted(
            self.key_bindings,
            key=lambda x: len(x['keys']),
            reverse=True
        )

        # Проходим по всем привязкам
        for binding in sorted_bindings:
            keys = binding['keys']

            # Если одна клавиша
            if len(keys) == 1:
                if key == keys[0]:
                    binding['action']()
                    return  # Прекращаем обработку после первого совпадения

            # Если комбинация клавиш (например, ctrl+c)
            else:
                # Проверяем, что нажата одна из клавиш комбинации
                if key in keys:
                    # Проверяем, что все остальные клавиши комбинации тоже нажаты
                    all_pressed = all(
                        held_keys.get(k, False) or key == k
                        for k in keys
                    )
                    if all_pressed:
                        binding['action']()
                        return

    def print_bindings(self):
        """Выводит все зарегистрированные привязки клавиш"""
        print("\n=== Привязки клавиш ===")
        for binding in self.key_bindings:
            keys_str = " + ".join(binding['keys'])
            print(f"{keys_str:20} - {binding['description']}")
        print("=" * 50)

    # ===== Действия по умолчанию =====

    def _quit_app(self):
        """Exit application"""
        print("Exiting application...")
        application.quit()

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        window.fullscreen = not window.fullscreen
        status = "enabled" if window.fullscreen else "disabled"
        print(f"Fullscreen {status}")

    def _toggle_mouse_lock(self):
        """Toggle mouse cursor lock"""
        mouse.locked = not mouse.locked
        status = "locked" if mouse.locked else "unlocked"
        print(f"Cursor {status}")

    def _print_camera_position(self):
        """Print camera position info"""
        if hasattr(camera, 'world_position'):
            print(f"Camera position: {camera.world_position}")
            print(f"Camera rotation: {camera.rotation}")

    def _toggle_freeze(self):
        """Toggle cursor freeze"""
        self.cursor_locked = not self.cursor_locked
        mouse.locked = self.cursor_locked
        mouse.visible = not self.cursor_locked

        # Блокируем/разблокируем player
        if self.player:
            self.player.enabled = self.cursor_locked

        status = "locked" if self.cursor_locked else "unlocked"
        print(f"Cursor freeze {status}")
    
    def _reset_all(self):
        """Reset zoom and player position"""
        # Сбрасываем зум
        if self.zoom_manager:
            self.zoom_manager.reset_zoom()
        
        # Сбрасываем позицию игрока
        if self.player and hasattr(self.player, 'reset_position'):
            self.player.reset_position()


# Глобальный обработчик ввода для Ursina
def input(key):
    """
    Глобальная функция для обработки ввода в Ursina
    Вызывается автоматически при нажатии клавиш
    """
    # Получаем глобальный input_manager из main.py
    if hasattr(application, 'input_manager'):
        application.input_manager.handle_input(key)
