# InputManager - Документация

## Обзор

InputManager - это простая и гибкая система управления вводом для Ursina, основанная на списке словарей с привязками клавиш.

## Основная концепция

Каждая привязка клавиши - это словарь с тремя полями:
- `keys`: одна клавиша (str) или несколько клавиш (tuple) для комбинаций
- `description`: описание действия
- `action`: функция, которая будет вызвана

## Базовое использование

```python
from input_manager import InputManager
from zoom_manager import ZoomManager

# Создаём ZoomManager
zoom_manager = ZoomManager(player=player)

# Создаём InputManager с автоматической регистрацией привязок для ZoomManager
input_manager = InputManager(zoom_manager=zoom_manager)

# Или создаём без ZoomManager и добавляем позже
input_manager = InputManager()
input_manager.set_zoom_manager(zoom_manager)

# Регистрируем свою привязку
input_manager.register_binding(
    keys='space',
    description="Прыжок",
    action=lambda: print("Прыжок!")
)

# Делаем доступным через application
app.input_manager = input_manager
```

## Автоматические привязки

InputManager автоматически регистрирует привязки для:

### Базовые функции (всегда)
- `Q` или `Escape` - Выход из приложения
- `F` - Переключить полноэкранный режим
- `M` - Показать/скрыть курсор
- `P` - Показать позицию камеры (debug)

### ZoomManager (если передан)
- `Scroll Up` - Приблизить (zoom in)
- `Scroll Down` - Отдалить (zoom out)
- `R` - Сбросить масштаб
- `1` - Увеличить размер объектов
- `2` - Уменьшить размер объектов

## Типы привязок

### 1. Одиночная клавиша

```python
def spawn_cube():
    print("Создан куб!")

input_manager.register_binding(
    keys='c',
    description="Создать куб",
    action=spawn_cube
)
```

### 2. Альтернативные клавиши (одно действие - несколько клавиш)

```python
input_manager.register_binding(
    keys=('q', 'escape'),  # Q или Escape
    description="Выход",
    action=quit_game
)
```

### 3. Комбинации клавиш (Ctrl+S, Alt+F4, и т.д.)

```python
input_manager.register_binding(
    keys=('left control', 's'),
    description="Сохранить",
    action=save_game
)

input_manager.register_binding(
    keys=('left shift', 'space'),
    description="Супер прыжок",
    action=super_jump
)
```

## Приоритет обработки

InputManager обрабатывает нажатия в следующем порядке:

1. **Сначала комбинации** (привязки с наибольшим количеством клавиш)
2. **Потом одиночные клавиши**
3. **Первое совпадение останавливает обработку** (return после выполнения действия)

Это означает, что если у вас есть:
- `Ctrl+S` → сохранить файл
- `S` → стрелять

То при нажатии `Ctrl+S` выполнится только сохранение, а не стрельба.

## Стандартные привязки

По умолчанию InputManager регистрирует следующие привязки:

| Клавиши | Действие |
|---------|----------|
| Q или Escape | Выход из приложения |
| F | Переключить полноэкранный режим |
| M | Показать/скрыть курсор |
| P | Показать позицию камеры (debug) |

## Утилиты

### Показать все привязки

```python
input_manager.print_bindings()
```

Выведет:
```
=== Привязки клавиш ===
q + escape          - Выход из приложения
f                   - Переключить полноэкранный режим
m                   - Показать/скрыть курсор
p                   - Показать позицию камеры
==================================================
```

## Полный пример

```python
from ursina import *
from input_manager import InputManager

app = Ursina()
input_manager = InputManager()

# Счётчик
counter = [0]

# Привязка 1: Space - увеличить
input_manager.register_binding(
    keys='space',
    description="Увеличить счётчик",
    action=lambda: counter.__setitem__(0, counter[0] + 1)
)

# Привязка 2: Ctrl+R - сброс
input_manager.register_binding(
    keys=('left control', 'r'),
    description="Сбросить счётчик",
    action=lambda: counter.__setitem__(0, 0)
)

# Привязка 3: H - показать счётчик
input_manager.register_binding(
    keys='h',
    description="Показать счётчик",
    action=lambda: print(f"Счётчик: {counter[0]}")
)

app.input_manager = input_manager
app.run()
```

## Интеграция с классами

Рекомендуемый способ интеграции с вашими классами:

```python
class Player:
    def __init__(self):
        self.speed = 5

    def move_forward(self):
        self.position += self.forward * self.speed * time.dt

    def shoot(self):
        Bullet(position=self.position)

# Создаём игрока
player = Player()

# Регистрируем привязки
input_manager.register_binding(
    keys='w',
    description="Движение вперёд",
    action=player.move_forward
)

input_manager.register_binding(
    keys='space',
    description="Выстрел",
    action=player.shoot
)
```

## Преимущества этого подхода

1. **Читаемость**: Все привязки видны в одном месте
2. **Гибкость**: Легко добавлять/удалять привязки
3. **Документированность**: Каждая привязка имеет описание
4. **Приоритеты**: Автоматическая обработка комбинаций перед одиночными клавишами
5. **Переиспользуемость**: Можно создавать разные наборы привязок для разных режимов

## Расширенные возможности

### Динамическое добавление привязок

```python
# Режим редактора
def enter_edit_mode():
    input_manager.register_binding(
        keys='e',
        description="Редактировать объект",
        action=edit_object
    )

# Выход из режима редактора (нужно реализовать удаление)
def exit_edit_mode():
    # Удалить привязку 'e'
    input_manager.key_bindings = [
        b for b in input_manager.key_bindings
        if 'e' not in b['keys']
    ]
```

### Условные привязки

```python
game_state = {'mode': 'normal'}

def context_sensitive_action():
    if game_state['mode'] == 'normal':
        print("Нормальное действие")
    elif game_state['mode'] == 'combat':
        print("Боевое действие")

input_manager.register_binding(
    keys='space',
    description="Контекстное действие",
    action=context_sensitive_action
)
```

## Известные ограничения

1. Комбинации работают только при использовании `held_keys` (нужно удерживать модификатор)
2. Порядок клавиш в кортеже не важен для комбинаций
3. Максимум проверяется только наличие всех клавиш, не последовательность нажатия
