# 🗺️ План развития архитектуры визуализации

**Дата создания:** 17 декабря 2025  
**Цель:** Постепенный переход к архитектуре, поддерживающей параллельную визуализацию TD3 vs CALF с множественными агентами

---

## 📋 Краткий план (чекбоксы)

### Фаза 0: Подготовка
- [ ] 0.1. Переименовать папку `math` → `physics` (избежать конфликта с встроенным модулем)
- [ ] 0.2. Убрать хак с `importlib` для импорта модуля
- [ ] 0.3. Запустить и проверить, что все работает как раньше

### Фаза 1: Упрощение менеджеров
- [ ] 1.1. Убрать `VisualsUpdateManager` (лишний слой)
- [ ] 1.2. Объединить `MathUpdateManager` в `SimulationEngine`
- [ ] 1.3. Упростить инициализацию (меньше зависимостей)
- [ ] 1.4. Проверить работу одной точки

### Фаза 2: Разделение математики и визуализации
- [ ] 2.1. Создать `StateBuffer` (пока простой dict)
- [ ] 2.2. `SimulationEngine` пишет состояния в буфер
- [ ] 2.3. `RenderEngine` читает состояния из буфера
- [ ] 2.4. Проверить работу с одной точкой через буфер

### Фаза 3: Абстракция Policy
- [ ] 3.1. Создать базовый класс `Policy`
- [ ] 3.2. Реализовать `PDPolicy` (простой контроллер)
- [ ] 3.3. Реализовать заглушку `TD3Policy` (пока случайные действия)
- [ ] 3.4. Проверить переключение между политиками

### Фаза 4: Векторизованные среды
- [ ] 4.1. Создать `VectorizedEnvironment` для N параллельных симуляций
- [ ] 4.2. Запустить 10 точек с PD контроллером
- [ ] 4.3. Запустить 50 точек с PD контроллером
- [ ] 4.4. Оценить производительность

### Фаза 5: Простые траектории
- [ ] 5.1. Создать `SimpleTrail` (одноцветная траектория)
- [ ] 5.2. Визуализировать траектории для 10 точек
- [ ] 5.3. Добавить decimation (скважность)
- [ ] 5.4. Визуализировать траектории для 50 точек
- [ ] 5.5. Добавить сброс траекторий по завершению эпизода

### Фаза 6: Dual визуализация (TD3 vs PD)
- [ ] 6.1. Создать две группы точек (TD3 слева, PD справа)
- [ ] 6.2. Синхронизировать начальные условия (одинаковый seed)
- [ ] 6.3. Визуализировать обе группы одновременно
- [ ] 6.4. Добавить статистику сравнения

### Фаза 7: CALF политика (3 режима)
- [ ] 7.1. Реализовать `CALFPolicy` с заглушками для режимов
- [ ] 7.2. Добавить переключение TD3/Fallback на основе простого условия
- [ ] 7.3. Проверить переключение на одной точке
- [ ] 7.4. Добавить третий режим Relax
- [ ] 7.5. Протестировать на 10 точках

### Фаза 8: Мультицветные траектории
- [ ] 8.1. Создать `MultiColorTrail` с группировкой по режимам
- [ ] 8.2. Визуализировать переключения на одной точке
- [ ] 8.3. Визуализировать 10 точек с мультицветными траекториями
- [ ] 8.4. Визуализировать 50 точек с мультицветными траекториями

### Фаза 9: Интеграция TD3 агента
- [ ] 9.1. Загрузить обученного TD3 агента
- [ ] 9.2. Подключить TD3 inference в `TD3Policy`
- [ ] 9.3. Протестировать на одной точке
- [ ] 9.4. Batch inference для множества точек
- [ ] 9.5. Dual визуализация: чистый TD3 vs CALF

### Фаза 10: Обучение с визуализацией
- [ ] 10.1. Режим headless (без визуализации) для быстрого обучения
- [ ] 10.2. Режим визуализации во время обучения (медленно)
- [ ] 10.3. Логирование и сравнение метрик
- [ ] 10.4. Сохранение checkpoints

### Фаза 11: Продвинутые фичи
- [ ] 11.1. Y-координата для визуализации награды
- [ ] 11.2. Регулировка порогов в runtime
- [ ] 11.3. Запись видео траекторий
- [ ] 11.4. Интерактивная камера и UI

### Фаза 12: Многопоточность (опционально)
- [ ] 12.1. Thread-safe `StateBuffer` (queue)
- [ ] 12.2. Симуляция в отдельном потоке
- [ ] 12.3. Рендер в главном потоке (Ursina)
- [ ] 12.4. Тестирование производительности

---

## 📖 Подробное описание фаз

### Фаза 0: Подготовка

**Цель:** Устранить технический долг перед рефакторингом

#### 0.1. Переименовать папку `math` → `physics`
**Что делать:**
- Переименовать `ursina/math/` → `ursina/physics/`
- Обновить все импорты в коде
- Убрать хак с `importlib.util`

**Почему:**
- Конфликт с встроенным модулем `math` Python
- Требует сложный workaround через `importlib`
- Имя `physics` более точно отражает содержимое (физические системы)

**Проверка:**
```bash
cd ursina
python main.py
# Должно запуститься без ошибок
```

#### 0.2. Убрать хак с `importlib`
**Что делать:**
```python
# Было:
import importlib.util
math_module_path = Path(__file__).parent / "math" / "__init__.py"
spec = importlib.util.spec_from_file_location("math_module", math_module_path)
# ...

# Станет:
from physics import PointSystem, MathUpdateManager
from physics.controllers import RotorController
```

**Проверка:**
- Все импорты работают
- Нет конфликтов имен

#### 0.3. Запустить и проверить работу
**Что проверять:**
- Приложение запускается
- Точка отображается
- Контроллер работает (точка двигается)
- Управление камерой работает (WASD, зум)

---

### Фаза 1: Упрощение менеджеров

**Цель:** Уменьшить количество менеджеров и упростить зависимости

#### 1.1. Убрать `VisualsUpdateManager`
**Проблема:**
`VisualsUpdateManager` - это просто обертка, которая вызывает `update()` у других менеджеров. Лишний слой абстракции.

**Что делать:**
```python
# Было в main.py:
def update():
    general_object_manager.update_all()
    visuals_update_manager.update_all()  # <- убираем этот слой

# Станет:
def update():
    general_object_manager.update_all()
    ui_manager.update()
    input_manager.update()
    zoom_manager.update()
    object_manager.update()
```

**Файлы для изменения:**
- `main.py` - убрать создание `VisualsUpdateManager`
- `main.py` - вызывать менеджеры напрямую
- `managers/visuals_update_manager.py` - можно удалить (или оставить для истории)

**Проверка:**
- Приложение работает так же
- Все менеджеры обновляются
- UI отображается

#### 1.2. Объединить `MathUpdateManager` в `SimulationEngine`
**Проблема:**
Два менеджера делают одно и то же:
- `MathUpdateManager` - хранит math объекты и вызывает `step()`
- `GeneralObjectManager` - тоже хранит math объекты и вызывает `step()`

**Что делать:**
1. Переименовать `MathUpdateManager` → `SimulationEngine`
2. Убрать дублирование с `GeneralObjectManager`
3. `SimulationEngine` отвечает только за математику
4. `GeneralObjectManager` связывает математику с визуализацией

**Структура:**
```
SimulationEngine (математика):
  - PointSystem objects
  - Controllers
  - step() для всех объектов

GeneralObjectManager (связь math↔visual):
  - Создает объекты в SimulationEngine
  - Создает визуальные объекты
  - Синхронизирует состояния
```

**Проверка:**
- Симуляция работает
- Нет дублирования вызовов `step()`

#### 1.3. Упростить инициализацию
**Цель:** Уменьшить количество зависимостей между менеджерами

**Было (10 менеджеров с жесткими зависимостями):**
```python
color_manager = ColorManager()
player = Player()
window_manager = WindowManager(color_manager, monitor="left")
zoom_manager = ZoomManager(player)
object_manager = ObjectManager(zoom_manager)
input_manager = InputManager(zoom_manager, player)
ui_manager = UIManager(color_manager, player, zoom_manager)
math_update_manager = MathUpdateManager()
general_object_manager = GeneralObjectManager(math_update_manager, object_manager, zoom_manager)
visuals_update_manager = VisualsUpdateManager(ui_manager, input_manager, zoom_manager, object_manager, general_object_manager)
```

**Станет (меньше зависимостей):**
```python
# Базовые компоненты
player = Player()
color_manager = ColorManager()

# Менеджеры (независимые где возможно)
window_manager = WindowManager(color_manager, monitor="left")
zoom_manager = ZoomManager(player)
input_manager = InputManager(zoom_manager, player)
object_manager = ObjectManager(zoom_manager)
ui_manager = UIManager(color_manager, player, zoom_manager)

# Симуляция
simulation_engine = SimulationEngine()
scene_manager = SceneManager(simulation_engine, object_manager)
```

**Проверка:**
- Порядок создания менее критичен
- Можно создавать менеджеры независимо

#### 1.4. Проверить работу одной точки
**Что проверять:**
- Точка создается
- Точка двигается по физике
- Контроллер влияет на движение
- Визуализация синхронизирована с математикой

---

### Фаза 2: Разделение математики и визуализации

**Цель:** Подготовить архитектуру к многопоточности через буфер состояний

#### 2.1. Создать `StateBuffer`
**Создать файл:** `ursina/core/state_buffer.py`

```python
class StateBuffer:
    """
    Буфер для передачи состояний от симуляции к визуализации.
    Пока простая реализация (dict), потом станет thread-safe.
    """
    def __init__(self):
        self._states = {}  # {obj_id: np.array}
    
    def write(self, obj_id: str, state: np.ndarray):
        """Записать состояние объекта (из симуляции)"""
        self._states[obj_id] = state.copy()
    
    def read(self, obj_id: str) -> np.ndarray:
        """Прочитать состояние объекта (для визуализации)"""
        return self._states.get(obj_id)
    
    def read_all(self) -> dict:
        """Прочитать все состояния"""
        return self._states.copy()
```

**Проверка:**
```python
# Тест
buffer = StateBuffer()
buffer.write('obj1', np.array([1.0, 2.0, 3.0]))
state = buffer.read('obj1')
print(state)  # [1.0, 2.0, 3.0]
```

#### 2.2. SimulationEngine пишет в буфер
**Что делать:**
```python
class SimulationEngine:
    def __init__(self, state_buffer=None):
        self.objects = {}
        self.state_buffer = state_buffer  # опционально
    
    def step(self):
        for obj_id, obj in self.objects.items():
            obj.step()
            
            # Если есть буфер - пишем состояние
            if self.state_buffer:
                state = obj.get_state()
                self.state_buffer.write(obj_id, state)
```

**Проверка:**
- Симуляция работает без буфера (обратная совместимость)
- Симуляция пишет в буфер, если он предоставлен

#### 2.3. RenderEngine читает из буфера
**Что делать:**
```python
class RenderEngine:
    def __init__(self, state_buffer):
        self.state_buffer = state_buffer
        self.visuals = {}
    
    def update(self):
        """Обновить визуализацию из буфера"""
        states = self.state_buffer.read_all()
        for obj_id, state in states.items():
            if obj_id in self.visuals:
                self.visuals[obj_id].update_from_state(state)
```

**Проверка:**
- Визуализация читает из буфера
- Состояния синхронизированы

#### 2.4. Проверить работу через буфер
**Интеграционный тест:**
```python
# Создаем буфер
buffer = StateBuffer()

# Симуляция пишет
sim = SimulationEngine(state_buffer=buffer)
sim.add_object('point1', PointSystem(...))

# Визуализация читает
render = RenderEngine(state_buffer=buffer)
render.add_visual('point1', PointVisual(...))

# Цикл
def update():
    sim.step()       # Обновили математику → записали в буфер
    render.update()  # Прочитали из буфера → обновили визуализацию
```

**Проверка:**
- Точка двигается
- Визуализация синхронизирована
- Можно отключить визуализацию (только симуляция)

---

### Фаза 3: Абстракция Policy

**Цель:** Создать интерфейс для разных политик (TD3, PD, CALF)

#### 3.1. Создать базовый класс `Policy`
**Создать файл:** `ursina/physics/policies/base_policy.py`

```python
from abc import ABC, abstractmethod
import numpy as np

class Policy(ABC):
    """Базовый класс для всех политик"""
    
    @abstractmethod
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Получить действие для одного состояния
        
        Parameters:
        -----------
        state : np.ndarray
            Состояние системы
        
        Returns:
        --------
        action : np.ndarray
            Действие
        """
        pass
    
    def get_actions_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Получить действия для батча состояний
        По умолчанию - цикл, но можно переопределить для векторизации
        """
        return np.array([self.get_action(s) for s in states])
```

**Проверка:**
- Класс создается
- Можно наследоваться

#### 3.2. Реализовать `PDPolicy`
**Создать файл:** `ursina/physics/policies/pd_policy.py`

```python
class PDPolicy(Policy):
    """PD контроллер"""
    def __init__(self, kp=1.0, kd=0.5, target=None):
        self.kp = kp
        self.kd = kd
        self.target = target if target is not None else np.zeros(2)
    
    def get_action(self, state):
        """
        state = [x, y, vx, vy, ...]
        action = Kp * error + Kd * error_dot
        """
        position = state[:2]
        velocity = state[2:4] if len(state) >= 4 else np.zeros(2)
        
        error = self.target - position
        error_dot = -velocity  # предполагаем, что target статичен
        
        action = self.kp * error + self.kd * error_dot
        return action
```

**Проверка:**
```python
policy = PDPolicy(kp=1.0, kd=0.5, target=np.array([1.0, 1.0]))
state = np.array([0.0, 0.0, 0.1, 0.1])
action = policy.get_action(state)
print(action)  # должно быть направлено к target
```

#### 3.3. Реализовать заглушку `TD3Policy`
**Создать файл:** `ursina/physics/policies/td3_policy.py`

```python
class TD3Policy(Policy):
    """TD3 агент (пока заглушка)"""
    def __init__(self, agent=None, action_dim=2):
        self.agent = agent
        self.action_dim = action_dim
    
    def get_action(self, state):
        if self.agent is None:
            # Заглушка: случайные действия
            return np.random.randn(self.action_dim) * 0.1
        else:
            # Реальный агент (Фаза 9)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = self.agent.actor(state_tensor)
            return action.cpu().numpy().squeeze()
```

**Проверка:**
```python
policy = TD3Policy(action_dim=2)
state = np.array([0.0, 0.0])
action = policy.get_action(state)
print(action)  # случайное действие
```

#### 3.4. Проверить переключение между политиками
**Тест:**
```python
# Создаем две политики
pd_policy = PDPolicy(kp=1.0, kd=0.5)
td3_policy = TD3Policy(action_dim=2)

# Создаем точку с PD
point1 = PointSystem(policy=pd_policy)

# Меняем политику в runtime
point1.policy = td3_policy

# Проверяем, что работает
state = point1.get_state()
action = point1.policy.get_action(state)
```

**Проверка:**
- Можно переключать политики
- Обе политики работают с одной системой

---

### Фаза 4: Векторизованные среды

**Цель:** Запустить N параллельных симуляций для множества точек

#### 4.1. Создать `VectorizedEnvironment`
**Создать файл:** `ursina/physics/vectorized_env.py`

```python
class VectorizedEnvironment:
    """N параллельных симуляций"""
    def __init__(self, n_envs, env_config, policy, seed=None):
        self.n_envs = n_envs
        self.envs = [create_environment(env_config) for _ in range(n_envs)]
        self.policy = policy
        
        # Векторизованные состояния
        self.states = None
        
        if seed is not None:
            for i, env in enumerate(self.envs):
                env.seed(seed + i)
    
    def reset(self):
        """Сброс всех сред"""
        self.states = np.array([env.reset() for env in self.envs])
        return self.states
    
    def step(self):
        """Batch шаг всех сред"""
        # Получаем действия от политики (batch)
        actions = self.policy.get_actions_batch(self.states)
        
        # Шаг каждой среды
        next_states = []
        rewards = []
        dones = []
        
        for i, env in enumerate(self.envs):
            s, r, d = env.step(actions[i])
            next_states.append(s)
            rewards.append(r)
            dones.append(d)
        
        self.states = np.array(next_states)
        return self.states, np.array(rewards), np.array(dones)
    
    def get_states(self):
        """Все состояния для визуализации"""
        return self.states.copy()
```

**Проверка:**
```python
policy = PDPolicy(kp=1.0, kd=0.5)
vec_env = VectorizedEnvironment(n_envs=10, env_config={}, policy=policy)
states = vec_env.reset()
print(states.shape)  # (10, state_dim)

for _ in range(100):
    states, rewards, dones = vec_env.step()
```

#### 4.2. Запустить 10 точек с PD
**Тест в main.py:**
```python
pd_policy = PDPolicy(kp=1.0, kd=0.5, target=np.array([1.0, 1.0]))
vec_env = VectorizedEnvironment(n_envs=10, env_config={}, policy=pd_policy)
vec_env.reset()

# Визуализация 10 точек
points = [PointVisual(color=color.red) for _ in range(10)]

def update():
    states, _, _ = vec_env.step()
    for i, state in enumerate(states):
        points[i].position = Vec3(state[0], 0, state[1])
```

**Проверка:**
- 10 точек отображаются
- Все двигаются к target
- FPS приемлемый

#### 4.3. Запустить 50 точек
**Тест:**
- То же самое, но `n_envs=50`
- Проверить производительность

**Проверка:**
- 50 точек работают
- FPS > 30

#### 4.4. Оценить производительность
**Метрики:**
- FPS для 10, 50, 100 точек
- Время на шаг симуляции
- Время на рендеринг

**Оптимизация при необходимости:**
- Векторизация физики (NumPy операции вместо цикла)
- Batch создание визуальных объектов

---

### Фаза 5: Простые траектории

**Цель:** Добавить визуализацию траекторий (хвостов)

#### 5.1. Создать `SimpleTrail`
**Создать файл:** `ursina/visuals/trail.py`

```python
class SimpleTrail:
    """Простая одноцветная траектория"""
    def __init__(self, color, max_length=200, decimation=1):
        self.color = color
        self.max_length = max_length
        self.decimation = decimation
        self.positions = []
        self.trail_entity = None
        self.step_counter = 0
    
    def add_point(self, position):
        """Добавить точку в траекторию"""
        self.step_counter += 1
        if self.step_counter % self.decimation != 0:
            return
        
        self.positions.append(Vec3(*position))
        if len(self.positions) > self.max_length:
            self.positions.pop(0)
        
        self.rebuild()
    
    def rebuild(self):
        """Перестроить визуализацию"""
        if self.trail_entity:
            destroy(self.trail_entity)
        
        if len(self.positions) >= 2:
            self.trail_entity = Entity(
                model=Mesh(vertices=self.positions, mode='line', thickness=2),
                color=self.color,
                alpha=0.7
            )
    
    def clear(self):
        """Очистить траекторию"""
        self.positions = []
        self.step_counter = 0
        if self.trail_entity:
            destroy(self.trail_entity)
```

**Проверка:**
```python
trail = SimpleTrail(color=color.red, max_length=100)
for i in range(200):
    trail.add_point([i*0.01, np.sin(i*0.1), 0])
# Должна появиться синусоида
```

#### 5.2. Визуализировать 10 точек с траекториями
**Тест:**
```python
n_points = 10
vec_env = VectorizedEnvironment(n_envs=n_points, ...)
trails = [SimpleTrail(color=color.red) for _ in range(n_points)]

def update():
    states, _, dones = vec_env.step()
    for i, state in enumerate(states):
        position = [state[0], 0, state[1]]
        trails[i].add_point(position)
        
        # Сброс при завершении эпизода
        if dones[i]:
            vec_env.envs[i].reset()
            trails[i].clear()
```

**Проверка:**
- 10 траекторий отображаются
- Траектории очищаются при сбросе

#### 5.3. Добавить decimation
**Тест:**
```python
# decimation=1: каждая точка
# decimation=2: каждая вторая
# decimation=5: каждая пятая

trail = SimpleTrail(color=color.red, decimation=2)
```

**Проверка:**
- Меньше точек в траектории
- Выше FPS
- Траектория все еще читаемая

#### 5.4. Визуализировать 50 точек
**Тест:**
- `n_points=50`
- `decimation=2` или `decimation=3`

**Проверка:**
- 50 траекторий работают
- FPS приемлемый

#### 5.5. Сброс траекторий по завершению эпизода
**Логика:**
```python
if done[i]:
    env.envs[i].reset()
    trails[i].clear()
```

**Проверка:**
- Траектории очищаются
- Новые траектории начинаются с начала

---

### Фаза 6: Dual визуализация (TD3 vs PD)

**Цель:** Сравнить две политики визуально

#### 6.1. Создать две группы точек
**Структура:**
```python
# Группа 1: TD3 (слева)
vec_env_td3 = VectorizedEnvironment(n_envs=50, policy=td3_policy, seed=42)
trails_td3 = [SimpleTrail(color=color.red) for _ in range(50)]

# Группа 2: PD (справа, сдвиг по Z)
vec_env_pd = VectorizedEnvironment(n_envs=50, policy=pd_policy, seed=42)
trails_pd = [SimpleTrail(color=color.green) for _ in range(50)]

def update():
    # TD3
    states_td3, _, dones_td3 = vec_env_td3.step()
    for i, state in enumerate(states_td3):
        pos = [state[0], 0, state[1]]
        trails_td3[i].add_point(pos)
    
    # PD (сдвиг по Z)
    states_pd, _, dones_pd = vec_env_pd.step()
    for i, state in enumerate(states_pd):
        pos = [state[0], 0, state[1] + 5]  # сдвиг!
        trails_pd[i].add_point(pos)
```

**Проверка:**
- Две группы по 50 точек
- Слева - красные (TD3)
- Справа - зеленые (PD)

#### 6.2. Синхронизировать начальные условия
**Важно:** Одинаковый seed для честного сравнения

```python
vec_env_td3 = VectorizedEnvironment(..., seed=42)
vec_env_pd = VectorizedEnvironment(..., seed=42)

# Проверка:
assert np.allclose(vec_env_td3.states, vec_env_pd.states)
```

**Проверка:**
- Обе группы начинают с одинаковых позиций
- Различия только из-за политик

#### 6.3. Визуализировать обе группы
**Проверка:**
- Видны различия в поведении
- PD стабильный (если настроен)
- TD3 пока случайный (заглушка)

#### 6.4. Добавить статистику
**UI элемент:**
```python
stats_text = Text(
    text='',
    position=(-0.85, 0.45),
    scale=1.2
)

def update_stats():
    n_success_td3 = sum(rewards_td3 > 0)
    n_success_pd = sum(rewards_pd > 0)
    
    stats_text.text = f'''TD3: {n_success_td3}/50 success
PD:  {n_success_pd}/50 success'''
```

**Проверка:**
- Статистика отображается
- Цифры обновляются

---

### Фаза 7: CALF политика (3 режима)

**Цель:** Реализовать CALF с переключением TD3/Relax/Fallback

#### 7.1. Реализовать `CALFPolicy` с заглушками
**Создать файл:** `ursina/physics/policies/calf_policy.py`

```python
class CALFPolicy(Policy):
    """CALF: TD3 с переключением на Relax/Fallback"""
    def __init__(self, td3_policy, pd_policy, 
                 fallback_threshold=0.3, relax_threshold=0.6):
        self.td3 = td3_policy
        self.pd = pd_policy
        self.fallback_threshold = fallback_threshold
        self.relax_threshold = relax_threshold
        
        self.current_mode = 'td3'  # для визуализации
    
    def get_action(self, state):
        # Заглушка для safety metric (пока случайное)
        safety_metric = np.random.rand()
        
        if safety_metric < self.fallback_threshold:
            self.current_mode = 'fallback'
            return self.pd.get_action(state)
        elif safety_metric < self.relax_threshold:
            self.current_mode = 'relax'
            # Смесь TD3 и PD
            alpha = (safety_metric - self.fallback_threshold) / \
                    (self.relax_threshold - self.fallback_threshold)
            return alpha * self.td3.get_action(state) + \
                   (1 - alpha) * self.pd.get_action(state)
        else:
            self.current_mode = 'td3'
            return self.td3.get_action(state)
```

**Проверка:**
```python
calf = CALFPolicy(td3_policy, pd_policy)
action = calf.get_action(state)
print(calf.current_mode)  # 'td3', 'relax', или 'fallback'
```

#### 7.2. Добавить переключение на основе простого условия
**Пример:** Расстояние от цели

```python
def get_safety_metric(self, state):
    """Простая метрика: расстояние от target"""
    position = state[:2]
    distance = np.linalg.norm(position - self.target)
    
    # Чем дальше, тем опаснее (меньше safety)
    safety = 1.0 / (1.0 + distance)
    return safety
```

**Проверка:**
- Вблизи target: safety высокий → TD3
- Далеко от target: safety низкий → Fallback

#### 7.3. Проверить на одной точке
**Тест:**
```python
calf = CALFPolicy(td3_policy, pd_policy, target=np.array([1, 1]))
point = PointSystem(policy=calf)

for step in range(100):
    point.step()
    print(f"Step {step}: mode={calf.current_mode}, pos={point.get_state()[:2]}")
```

**Проверка:**
- Режимы переключаются
- При приближении к target: fallback → relax → td3

#### 7.4. Добавить третий режим Relax
**Уже реализовано в 7.1** - смесь TD3 и PD

**Проверка:**
- Три режима работают
- Плавный переход между режимами (через Relax)

#### 7.5. Протестировать на 10 точках
**Тест:**
```python
calf_policy = CALFPolicy(td3_policy, pd_policy)
vec_env = VectorizedEnvironment(n_envs=10, policy=calf_policy)

# Визуализация с цветовым кодированием
def update():
    states, _, _ = vec_env.step()
    for i, state in enumerate(states):
        mode = calf_policy.get_mode_for_env(i)  # нужно добавить batch
        color = MODE_COLORS[mode]
        points[i].color = color
```

**Проверка:**
- 10 точек с переключениями
- Цвета меняются в зависимости от режима

---

### Фаза 8: Мультицветные траектории

**Цель:** Визуализировать переключения CALF в траекториях

#### 8.1. Создать `MultiColorTrail`
**Файл:** `ursina/visuals/multi_color_trail.py`

```python
class MultiColorTrail:
    """Траектория с сегментами разного цвета"""
    
    MODE_COLORS = {
        'td3': color.blue,
        'relax': color.green,
        'fallback': color.orange
    }
    
    def __init__(self, max_length=200, decimation=1):
        self.max_length = max_length
        self.decimation = decimation
        self.positions = []
        self.modes = []
        self.segments = []
        self.step_counter = 0
    
    def add_point(self, position, mode):
        """Добавить точку с режимом"""
        self.step_counter += 1
        if self.step_counter % self.decimation != 0:
            return
        
        self.positions.append(Vec3(*position))
        self.modes.append(mode)
        
        if len(self.positions) > self.max_length:
            self.positions.pop(0)
            self.modes.pop(0)
        
        self.rebuild_trail()
    
    def rebuild_trail(self):
        """Перестроить с группировкой по режимам"""
        # Удаляем старые сегменты
        for seg in self.segments:
            destroy(seg)
        self.segments = []
        
        # Группируем по режимам
        groups = self._group_by_mode()
        
        # Рисуем каждую группу
        for mode, points in groups:
            if len(points) >= 2:
                seg = Entity(
                    model=Mesh(vertices=points, mode='line', thickness=3),
                    color=self.MODE_COLORS[mode],
                    alpha=0.7
                )
                self.segments.append(seg)
    
    def _group_by_mode(self):
        """Группировка последовательных точек по режиму"""
        # (см. подробную реализацию выше в коде)
        pass
```

**Проверка:**
```python
trail = MultiColorTrail()
trail.add_point([0, 0, 0], 'td3')
trail.add_point([0.1, 0, 0], 'td3')
trail.add_point([0.2, 0, 0], 'fallback')
trail.add_point([0.3, 0, 0], 'fallback')
# Должны быть синий и оранжевый сегменты
```

#### 8.2. Визуализировать переключения на одной точке
**Тест:**
```python
calf = CALFPolicy(td3, pd)
trail = MultiColorTrail()

def update():
    state = env.get_state()
    action = calf.get_action(state)
    env.step(action)
    
    position = [state[0], 0, state[1]]
    trail.add_point(position, calf.current_mode)
```

**Проверка:**
- Траектория меняет цвет при переключениях
- Синий (TD3) → Зеленый (Relax) → Оранжевый (Fallback)

#### 8.3. Визуализировать 10 точек
**Тест:**
- 10 точек с `MultiColorTrail`
- Каждая со своими переключениями

**Проверка:**
- Видны индивидуальные паттерны переключений
- Производительность приемлемая

#### 8.4. Визуализировать 50 точек
**Тест:**
- 50 точек с мультицветными траекториями
- Decimation=2 или 3

**Проверка:**
- FPS > 30
- Траектории читаемые
- Переключения видны

---

### Фаза 9: Интеграция TD3 агента

**Цель:** Подключить реального обученного TD3 агента

#### 9.1. Загрузить обученного TD3
**Предполагается:**
- Есть файл `calf_model.pth` или `td3_agent.pth`
- Есть класс TD3 агента

**Код:**
```python
from RL.td3 import TD3Agent

# Загрузка агента
agent = TD3Agent(state_dim=..., action_dim=..., max_action=...)
agent.load('RL/calf_model.pth')
agent.actor.eval()  # evaluation mode
```

**Проверка:**
```python
state = np.array([0, 0, 0, 0])
action = agent.select_action(state)
print(action)  # должно быть реалистичное действие
```

#### 9.2. Подключить в `TD3Policy`
**Обновить:** `ursina/physics/policies/td3_policy.py`

```python
class TD3Policy(Policy):
    def __init__(self, agent):
        self.agent = agent
    
    def get_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.agent.actor(state_tensor)
        return action.cpu().numpy().squeeze()
```

**Проверка:**
- Агент выдает действия
- Нет утечек памяти

#### 9.3. Протестировать на одной точке
**Тест:**
```python
td3_policy = TD3Policy(agent)
env = Environment()
state = env.reset()

for _ in range(100):
    action = td3_policy.get_action(state)
    state, reward, done = env.step(action)
    if done:
        break

print(f"Reward: {reward}")
```

**Проверка:**
- Агент управляет системой
- Поведение осмысленное (не случайное)

#### 9.4. Batch inference для множества точек
**Оптимизация:**
```python
def get_actions_batch(self, states):
    """Batch inference - быстрее чем цикл"""
    with torch.no_grad():
        states_tensor = torch.FloatTensor(states)  # (N, state_dim)
        actions = self.agent.actor(states_tensor)  # (N, action_dim)
    return actions.cpu().numpy()
```

**Проверка:**
```python
states = np.random.randn(50, state_dim)
actions = td3_policy.get_actions_batch(states)
print(actions.shape)  # (50, action_dim)
```

#### 9.5. Dual визуализация: чистый TD3 vs CALF
**Финальный тест Фазы 9:**
```python
# Загружаем агента
agent = TD3Agent(...)
agent.load('RL/calf_model.pth')

# Политики
td3_policy = TD3Policy(agent)
pd_policy = PDPolicy(kp=1.0, kd=0.5)
calf_policy = CALFPolicy(td3_policy, pd_policy)

# Dual визуализация
vis = DualVisualization(
    n_points=50,
    policy_1=td3_policy,
    policy_2=calf_policy,
    seed=42
)

def update():
    vis.step()
```

**Проверка:**
- Видны различия между чистым TD3 и CALF
- CALF переключается на fallback в критических ситуациях
- Статистика показывает улучшения

---

### Фаза 10: Обучение с визуализацией

**Цель:** Интеграция обучения TD3 с визуализацией

#### 10.1. Режим headless
**Флаг для отключения визуализации:**
```python
class SimulationConfig:
    def __init__(self, render_mode=None):
        self.render_mode = render_mode  # None, 'human', 'rgb_array'

# Быстрое обучение
sim = Simulation(render_mode=None)  # без визуализации

# С визуализацией
sim = Simulation(render_mode='human')
```

**Проверка:**
- Headless mode быстрее в 10+ раз
- Обучение работает без визуализации

#### 10.2. Режим визуализации во время обучения
**Опция:**
```python
# Обучение с визуализацией каждые N эпизодов
for episode in range(max_episodes):
    # Обучение без рендера (быстро)
    if episode % render_every != 0:
        sim.render_mode = None
    else:
        sim.render_mode = 'human'
    
    # Эпизод
    run_episode(sim, agent)
```

**Проверка:**
- Можно видеть прогресс
- Не сильно замедляет обучение

#### 10.3. Логирование и сравнение метрик
**Метрики:**
- Награда за эпизод
- Количество переключений на fallback
- Среднее расстояние до цели
- Успешность (% достижения цели)

**Визуализация:**
```python
import matplotlib.pyplot as plt

metrics = {
    'td3_rewards': [],
    'calf_rewards': [],
    'fallback_activations': []
}

# После эпизода
metrics['td3_rewards'].append(total_reward_td3)
metrics['calf_rewards'].append(total_reward_calf)

# График
plt.plot(metrics['td3_rewards'], label='TD3')
plt.plot(metrics['calf_rewards'], label='CALF')
plt.legend()
plt.savefig('training_comparison.png')
```

**Проверка:**
- Метрики логируются
- Графики показывают прогресс
- CALF не хуже TD3 (а желательно лучше!)

#### 10.4. Сохранение checkpoints
**Код:**
```python
# Сохранение агента
if episode % save_every == 0:
    agent.save(f'checkpoints/agent_episode_{episode}.pth')
    
    # Сохранение метрик
    np.savez(f'checkpoints/metrics_episode_{episode}.npz', **metrics)
```

**Проверка:**
- Checkpoints сохраняются
- Можно продолжить обучение
- Можно загрузить для визуализации

---

### Фаза 11: Продвинутые фичи

**Цель:** Улучшить визуализацию и интерактивность

#### 11.1. Y-координата для награды
**Модификация:**
```python
def state_to_position(self, state, reward=None):
    """3D позиция с Y = награда"""
    x, z = state[0], state[1]
    
    if self.use_reward_height and reward is not None:
        y = reward * self.reward_scale
    else:
        y = 0.0
    
    return [x, y, z]
```

**Проверка:**
- Траектории идут вверх при положительной награде
- Траектории идут вниз при отрицательной награде
- Визуально понятно, где система получает награды

#### 11.2. Регулировка порогов в runtime
**Интерактивное управление:**
```python
def input(key):
    if key == 'up arrow':
        calf_policy.fallback_threshold += 0.05
        print(f"Fallback threshold: {calf_policy.fallback_threshold:.2f}")
    
    if key == 'down arrow':
        calf_policy.fallback_threshold -= 0.05
    
    if key == 'right arrow':
        calf_policy.relax_threshold += 0.05
    
    if key == 'left arrow':
        calf_policy.relax_threshold -= 0.05
```

**Проверка:**
- Пороги меняются в реальном времени
- Видно влияние на поведение
- Можно найти оптимальные настройки визуально

#### 11.3. Запись видео траекторий
**Использование:**
```python
from ursina import Recorder

recorder = Recorder()

def input(key):
    if key == 'v':
        recorder.start_recording()
        print("Recording started...")
    
    if key == 'b':
        recorder.stop_recording()
        recorder.save('trajectory_video.mp4')
        print("Video saved!")
```

**Проверка:**
- Видео записывается
- Качество приемлемое
- Можно использовать для презентаций

#### 11.4. Интерактивная камера и UI
**Фичи:**
- Свободное вращение камеры
- Зум к конкретным точкам
- Toggle различных элементов
- Пауза/продолжение симуляции

**Код:**
```python
class CameraController:
    def __init__(self):
        self.paused = False
        self.follow_mode = None  # None или индекс точки
    
    def handle_input(self, key):
        if key == 'space':
            self.paused = not self.paused
        
        if key == 'f':
            # Follow random point
            self.follow_mode = np.random.randint(0, n_points)
        
        if key == 'escape':
            self.follow_mode = None
```

**Проверка:**
- Камера управляется удобно
- Можно следить за конкретными точками
- Можно паузить для анализа

---

### Фаза 12: Многопоточность (опционально)

**Цель:** Разделить симуляцию и рендер по разным потокам

**Внимание:** Ursina работает только в главном потоке! Нельзя создавать Entity в другом потоке.

**Правильный подход:**
- Симуляция в отдельном потоке (пишет в буфер)
- Рендер в главном потоке (читает из буфера)

#### 12.1. Thread-safe `StateBuffer`
**Обновить:** `ursina/core/state_buffer.py`

```python
import threading
from queue import Queue

class ThreadSafeStateBuffer:
    """Thread-safe буфер для многопоточности"""
    def __init__(self):
        self._lock = threading.Lock()
        self._states = {}
    
    def write(self, obj_id, state):
        """Вызывается из потока симуляции"""
        with self._lock:
            self._states[obj_id] = state.copy()
    
    def read_all(self):
        """Вызывается из главного потока (рендер)"""
        with self._lock:
            return self._states.copy()
```

**Проверка:**
```python
# Тест многопоточности
buffer = ThreadSafeStateBuffer()

def writer_thread():
    for i in range(1000):
        buffer.write('obj1', np.array([i, i*2]))

def reader_thread():
    for i in range(1000):
        states = buffer.read_all()
        time.sleep(0.001)

# Запуск
t1 = threading.Thread(target=writer_thread)
t2 = threading.Thread(target=reader_thread)
t1.start()
t2.start()
t1.join()
t2.join()
# Не должно быть race conditions
```

#### 12.2. Симуляция в отдельном потоке
**Код:**
```python
class SimulationThread:
    def __init__(self, simulation, state_buffer):
        self.simulation = simulation
        self.state_buffer = state_buffer
        self.running = False
        self.thread = None
    
    def start(self):
        """Запустить поток симуляции"""
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
    
    def _run(self):
        """Цикл симуляции (в отдельном потоке)"""
        while self.running:
            # Шаг симуляции
            self.simulation.step()
            
            # Запись состояний в буфер
            states = self.simulation.get_states()
            for obj_id, state in states.items():
                self.state_buffer.write(obj_id, state)
            
            # Можно добавить ограничение FPS
            # time.sleep(1/simulation_fps)
    
    def stop(self):
        """Остановить поток"""
        self.running = False
        if self.thread:
            self.thread.join()
```

**Проверка:**
```python
buffer = ThreadSafeStateBuffer()
sim_thread = SimulationThread(simulation, buffer)
sim_thread.start()

# Главный поток - рендер
def update():
    states = buffer.read_all()
    render_engine.update(states)

# При выходе
sim_thread.stop()
```

#### 12.3. Рендер в главном потоке
**Уже реализовано** - `update()` работает в главном потоке Ursina

**Проверка:**
- Симуляция бежит быстро (в своем потоке)
- Рендер обновляется с 60 FPS (пропуская кадры симуляции)
- Нет race conditions

#### 12.4. Тестирование производительности
**Метрики:**
- Simulation FPS (без рендера): должно быть >> 60
- Render FPS: 60
- Latency: задержка между симуляцией и отображением

**Сравнение:**
```
Однопоточный:
  Simulation: 60 FPS (ограничено рендером)
  Render: 60 FPS
  Total: 60 FPS

Многопоточный:
  Simulation: 500+ FPS (без рендера)
  Render: 60 FPS
  Total: симуляция быстрее в 8+ раз!
```

**Проверка:**
- Многопоточность дает ускорение
- Визуализация не тормозит обучение
- Можно запускать headless на сервере

---

## 🎯 Итоговая архитектура

После прохождения всех фаз:

```
┌─────────────────────────────────────────────────────┐
│                   Main Application                  │
│                                                     │
│  ┌─────────────────┐         ┌──────────────────┐ │
│  │ Simulation      │  State  │ Render Engine    │ │
│  │ Thread          │ ──────> │ (Main Thread)    │ │
│  │ (CPU)           │ Buffer  │ (Ursina)         │ │
│  │                 │         │                  │ │
│  │ - Physics       │         │ - Visuals        │ │
│  │ - TD3 Agent     │         │ - Trails         │ │
│  │ - CALF Logic    │         │ - UI             │ │
│  │ - 100+ envs     │         │ - Camera         │ │
│  └─────────────────┘         └──────────────────┘ │
│                                                     │
│  Policies:                   Components:           │
│  - TD3Policy                 - StateBuffer         │
│  - PDPolicy                  - VectorizedEnv       │
│  - CALFPolicy                - MultiColorTrail     │
└─────────────────────────────────────────────────────┘
```

**Ключевые принципы:**
1. ✅ Разделение математики и визуализации
2. ✅ Policy как интерфейс (легко менять/сравнивать)
3. ✅ StateBuffer для независимости компонентов
4. ✅ Векторизация для производительности
5. ✅ Постепенное развитие с тестированием

---

## 📝 Примечания

- **Каждую фазу тестируем отдельно** - не переходим дальше, пока не работает
- **Можно пропустить Фазу 12** (многопоточность) - она опциональная
- **Сохраняем старый код** - можно откатиться, если что-то сломается
- **Визуализация важна** - она помогает понять, что происходит в CALF

---

## 🚀 С чего начать?

**Следующий шаг:** Фаза 0.1 - переименовать `math` → `physics`

```bash
cd ursina
mv math physics
# Обновить импорты в файлах
```

Удачи! 🎯
