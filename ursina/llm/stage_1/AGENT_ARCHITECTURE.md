# Новая архитектура: Agent с интегрированной траекторией

## Проблема (ДО)

Траектории жили отдельно от агентов:
- В `main.py` создавались отдельные списки `points_td3`, `trails_td3`, `points_calf`, `trails_calf`
- Траектории обновлялись вручную в функции `update()`
- Управление траекториями было разделено от управления агентами
- Приходилось вручную синхронизировать индексы агентов и траекторий

```python
# СТАРЫЙ КОД (до рефакторинга)
points_td3 = []
trails_td3 = []

for i in range(N_AGENTS):
    # Создаём точку агента
    point = object_manager.create_object(...)
    points_td3.append(point)

    # Создаём траекторию
    trail = MultiColorTrail(max_length=600)
    trails_td3.append(trail)

# В update():
for i in range(len(points_td3)):
    # Обновляем физику
    vec_env.step()

    # Вручную обновляем визуализацию
    points_td3[i].position = new_position

    # Вручную обновляем траекторию
    trails_td3[i].add_point(new_position)
```

## Решение (ПОСЛЕ)

Каждый агент владеет своей траекторией (кольцевой буфер):

### 1. Класс Agent ([physics/agent.py](physics/agent.py))

```python
class Agent:
    """Агент с встроенной траекторией и визуализацией"""

    def __init__(self, point_system, object_manager, name, ...):
        # Физика
        self.point_system = point_system

        # Визуальное представление (шарик агента)
        self.visual = object_manager.create_object(...)

        # КОЛЬЦЕВОЙ БУФЕР для траектории
        # Создаём пул точек один раз
        self.trail_points = []
        for i in range(max_length):
            point = object_manager.create_object(
                model='sphere',
                scale=point_size,
                visible=False
            )
            self.trail_points.append(point)

        # Индексы кольцевого буфера
        self.trail_head = 0  # Куда писать следующую точку
        self.trail_count = 0  # Сколько точек активно
```

### 2. Принцип работы траектории (кольцевой буфер)

**При создании агента:**
- Создаётся фиксированный пул точек (например, 150 точек)
- Все точки скрыты и помещены за границы видимости
- Никаких `destroy()` в будущем не будет!

**При каждом шаге агента:**
```python
def update_position(self, state, mode):
    # 1. Обновляем визуализацию агента
    self.visual.position = new_position

    # 2. Добавляем точку в траекторию (кольцевой буфер)
    self._add_trail_point(new_position, mode)

def _add_trail_point(self, position, mode):
    # Decimation: добавляем только каждую N-ую точку
    if self.step_counter % self.decimation != 0:
        return

    # ПЕРЕИСПОЛЬЗУЕМ самую старую точку
    point = self.trail_points[self.trail_head]
    point.position = position
    point.color = MODE_COLORS[mode]
    point.visible = True

    # Сдвигаем head (кольцевой буфер)
    self.trail_head = (self.trail_head + 1) % len(self.trail_points)

    # Увеличиваем count до максимума
    if self.trail_count < len(self.trail_points):
        self.trail_count += 1
```

**Когда очередь заполняется:**
- `trail_head` возвращается к началу (0)
- Самая старая точка становится первой в очереди
- Она перемещается на новую позицию
- **НЕТ destroy/create** - только перемещение!

### 3. Интеграция в VectorizedEnvironment ([physics/vectorized_env.py](physics/vectorized_env.py))

```python
class VectorizedEnvironment:
    def __init__(
        self,
        ...,
        object_manager=None,
        create_agents=True,  # НОВЫЙ ПАРАМЕТР
        trail_config=None
    ):
        # Создаём N агентов с траекториями
        for i in range(n_envs):
            point_system = PointSystem(...)

            if create_agents:
                agent = Agent(
                    point_system=point_system,
                    object_manager=object_manager,
                    trail_config=trail_config
                )
                self.agents.append(agent)

    def step(self):
        # Получаем действия от политики
        actions = self.policy.get_actions_batch(self.states)

        # Обновляем агентов (физика + визуализация + траектория)
        for i, agent in enumerate(self.agents):
            # Применяем действие
            agent.point_system.u = actions[i]
            agent.point_system.step()

            # Получаем режим (для CALF)
            mode = self.policy.get_mode_for_env(i)

            # АВТОМАТИЧЕСКИ обновляется визуализация И траектория!
            agent.update_position(state, mode=mode)
```

### 4. Использование в main.py ([main.py](main.py))

```python
# НОВЫЙ КОД (после рефакторинга)

# Конфигурация траекторий
trail_config = {
    'max_length': 150,    # Размер кольцевого буфера
    'decimation': 3,      # Каждую 3-ю точку добавлять
    'point_size': 0.03    # Размер точек
}

# Создаём векторизованную среду с агентами
vec_env_td3 = VectorizedEnvironment(
    n_envs=15,
    policy=td3_policy,
    object_manager=object_manager,
    group_name="td3",
    offset=(-8, 0, 0),  # LEFT side
    color=Vec4(0.2, 0.3, 0.8, 1),
    trail_config=trail_config,
    create_agents=True  # ВКЛЮЧАЕМ создание Agent объектов
)

# Агенты уже созданы! Каждый владеет своей траекторией
print(f"Created {len(vec_env_td3.agents)} agents with trails")

def update():
    # VectorizedEnvironment.step() автоматически:
    # - Обновляет физику
    # - Обновляет визуализацию агентов
    # - Обновляет траектории (кольцевой буфер)
    vec_env_td3.step()

    # Только статистика и сброс при достижении цели
    for i in range(vec_env_td3.n_envs):
        state = vec_env_td3.envs[i].state
        distance = np.linalg.norm(state)

        if distance < 0.15:  # Достигли цели
            # reset_agent автоматически очищает траекторию
            vec_env_td3.reset_agent(i)
```

## Преимущества новой архитектуры

### 1. Эффективность памяти
- **ДО:** При каждом добавлении точки создавался новый объект, старые удалялись через `destroy()`
- **ПОСЛЕ:** Пул объектов создаётся один раз, затем только переиспользуется
- Экономия: для 30 агентов с траекториями по 150 точек = 4500 объектов создаются один раз вместо постоянного create/destroy

### 2. Производительность
- Нет вызовов `destroy()` и повторных `create()` каждый кадр
- Только обновление `position` и `color` существующих объектов
- Значительно меньше нагрузка на сборщик мусора Python

### 3. Инкапсуляция
- Агент владеет своей траекторией
- Траектория автоматически обновляется при `agent.update_position()`
- Не нужно вручную синхронизировать индексы агентов и траекторий

### 4. Удобство использования
- Простое создание: `VectorizedEnvironment(..., create_agents=True)`
- Автоматическое обновление: `vec_env.step()`
- Простой сброс: `vec_env.reset_agent(i)` автоматически очищает траекторию

### 5. Гибкость
- Можно настроить размер буфера (`max_length`)
- Decimation для контроля плотности точек
- Поддержка разных режимов (TD3, Relax, Fallback) с автоматической раскраской

## Файлы изменённые/созданные

1. **Новые файлы:**
   - [physics/agent.py](physics/agent.py) - класс Agent с интегрированной траекторией

2. **Изменённые файлы:**
   - [physics/vectorized_env.py](physics/vectorized_env.py) - добавлена поддержка создания Agent объектов
   - [physics/__init__.py](physics/__init__.py) - экспорт Agent
   - [main.py](main.py) - использование новой архитектуры

## Технические детали

### Кольцевой буфер (Ring Buffer)

```
Пул точек: [P0, P1, P2, P3, P4, ..., P149]
             ↑
             trail_head = 0
             trail_count = 0

Шаг 1: добавляем точку → P0 перемещается на новую позицию
         trail_head = 1, trail_count = 1

Шаг 2: добавляем точку → P1 перемещается на новую позицию
         trail_head = 2, trail_count = 2

...

Шаг 150: буфер заполнен
         trail_head = 0 (вернулся к началу!)
         trail_count = 150

Шаг 151: P0 (самая старая) перемещается на новую позицию
         trail_head = 1
         trail_count = 150 (не растёт дальше)
```

### Decimation

Чтобы не добавлять точку каждый кадр:
```python
decimation = 3  # Каждую 3-ю точку добавлять

step_counter = 0
if step_counter % decimation != 0:
    return  # Пропускаем этот кадр
```

При 60 FPS и decimation=3 → 20 точек в секунду
При буфере 150 точек → 7.5 секунд истории траектории

## Тестирование

Синтаксис проверен:
```bash
python -m py_compile physics/agent.py            # OK
python -m py_compile physics/vectorized_env.py   # OK
python -m py_compile main.py                     # OK
```

Готово к запуску с Ursina!
