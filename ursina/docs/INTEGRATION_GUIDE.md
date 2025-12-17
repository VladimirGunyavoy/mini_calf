# Руководство по интеграции SceneSetup

## Проблема

У вас есть два подхода к настройке сцены:

### Текущий в main.py (простой):
```python
player = Player()  # FirstPersonController
zoom_manager = ZoomManager(player=player)
```

### SceneSetup (сложный):
```python
scene_setup = SceneSetup()
# Включает в себя:
# - player (FirstPersonController)
# - lights
# - ui_manager
# - color_manager
# - управление курсором
# - toggle_freeze()
```

## Два варианта интеграции

### ❶ Вариант 1: Минимальная интеграция (РЕКОМЕНДУЮ)

**Плюсы:** Простота, не ломает существующий код
**Минусы:** Дублирование функционала

Оставляем текущий main.py как есть и добавляем только нужные части из SceneSetup:

```python
# main.py
from color_manager import ColorManager
from ui_manager import UIManager

# Менеджеры
color_manager = ColorManager()
ui_manager = UIManager(color_manager)

# Player уже есть
player = Player()

# ZoomManager уже есть
zoom_manager = ZoomManager(player=player)

# Опционально: добавить UI элементы
ui_manager.create_position_info()
ui_manager.create_instructions("instructions", "Нажмите H для справки")
```

### ❷ Вариант 2: Полная замена на SceneSetup

**Плюсы:** Всё в одном месте
**Минусы:** Нужно адаптировать SceneSetup под InputManager

```python
# main.py
from scene_setup import SceneSetup

# Создаём SceneSetup (включает player, lights, ui_manager, color_manager)
scene_setup = SceneSetup(
    init_position=(0, 5, -10),
    init_rotation_x=0,
    init_rotation_y=0
)

# Активируем режим InputManager
scene_setup.enable_input_manager_mode(True)

# Теперь player доступен через scene_setup.player
zoom_manager = ZoomManager(player=scene_setup.player)
```

## Что нужно для полной интеграции?

### Конфликты функционала:

1. **Player** - и в main.py и в SceneSetup
2. **Освещение** - есть setup.lighting() и SceneSetup.lights
3. **UI** - SceneSetup создаёт свой UIManager
4. **Курсор** - SceneSetup управляет через toggle_freeze()

### Решение конфликтов:

#### Способ А: Упростить SceneSetup
Убрать из SceneSetup создание player, оставить только:
- Освещение
- UIManager
- ColorManager
- Курсор

#### Способ Б: Использовать SceneSetup как единственный источник
Удалить из main.py:
- `player = Player()`
- `setup.lighting()`
- Использовать `scene_setup.player`

## Моя рекомендация

### Гибридный подход:

```python
# main.py
from color_manager import ColorManager
from ui_manager import UIManager

# 1. ColorManager (для цветов)
color_manager = ColorManager()

# 2. UIManager (для UI элементов)
ui_manager = UIManager(color_manager)

# 3. Player (как сейчас)
player = Player()

# 4. ZoomManager (как сейчас)
zoom_manager = ZoomManager(player=player)

# 5. InputManager (как сейчас)
input_manager = InputManager()

# 6. Опционально: SceneSetup только для специфичных функций
# Если нужен toggle_freeze или особое освещение
from scene_setup import SceneSetup
scene_setup = SceneSetup(...)
scene_setup.enable_input_manager_mode(True)
```

## Что добавить в текущий main.py прямо сейчас?

Минимум для старта:

```python
# В начало main.py
from color_manager import ColorManager
from ui_manager import UIManager

# После app = Ursina()
color_manager = ColorManager()
ui_manager = UIManager(color_manager)

# После создания player
# Опционально: UI элементы
position_info = ui_manager.create_position_info()

def update():
    # Обновляем позицию игрока в UI
    if player:
        ui_manager.update_text('main',
            f"Position: {player.position.x:.2f}, {player.position.y:.2f}, {player.position.z:.2f}")
```

## Следующие шаги

1. **Протестируйте минимальную интеграцию** (ColorManager + UIManager)
2. **Решите, нужен ли вам SceneSetup полностью** или только отдельные части
3. **Если нужен SceneSetup**, создадим упрощённую версию без конфликтов

Что вы хотите сделать?
- [ ] Минимальная интеграция (ColorManager + UIManager)
- [ ] Полная замена на SceneSetup
- [ ] Упростить SceneSetup под текущую архитектуру
