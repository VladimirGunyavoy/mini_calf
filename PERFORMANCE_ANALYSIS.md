# 🔬 Глубокий технический анализ проблемы производительности

**Дата**: 2025-12-18
**Файл**: `ursina/train_calf_visual.py`
**Проблема**: Резкое падение FPS по сравнению с `train_td3_visual.py`

---

## 📍 Локализация проблемы

**Файл**: [ursina/train_calf_visual.py:402](ursina/train_calf_visual.py#L402)
**Проблемный вызов**:
```python
vis_actions, vis_modes = calf_agent.select_action_batch(vis_states, exploration_noise=0.0, return_modes=True)
```

**Факт**: Метод `select_action_batch` **НЕ СУЩЕСТВУЕТ** в классе `CALFController` ([RL/calf.py](RL/calf.py))

**Доступные методы в CALFController**:
- `select_action(state, exploration_noise=0.0)` - только для одного агента
- НЕТ `select_action_batch()` - метод отсутствует!

---

## 🚨 Критические проблемы

### **Проблема 1: AttributeError при запуске**

Код **упадёт с ошибкой** при достижении `training_stats['training_started'] = True`:

```python
AttributeError: 'CALFController' object has no attribute 'select_action_batch'
```

**Когда произойдёт**: На шаге **1000** (константа `START_TRAINING_STEP = 1000`).

До этого момента выполняется блок `else` (строки 404-406) с рандомными действиями:
```python
vis_actions = np.random.uniform(-env.max_action, env.max_action, size=(len(visual_envs), env.action_dim))
vis_modes = ['td3'] * len(visual_envs)
```

---

### **Проблема 2: Множественные GPU вызовы (если был workaround)**

**Гипотеза**: Если пользователь где-то реализовал временный workaround (например, цикл), то производительность страдает из-за:

#### **Анализ GPU нагрузки при вызове `select_action` в цикле**

Если бы код выглядел так:
```python
vis_actions = []
vis_modes = []
for state in vis_states:  # 25 итераций
    action = calf_agent.select_action(state, exploration_noise=0.0)
    vis_actions.append(action)
    vis_modes.append('td3')  # placeholder
```

**Каждый вызов `select_action`** ([RL/calf.py:165](RL/calf.py#L165)) делает:

#### 1. **Actor inference** (строка 181)
```python
action_actor = self.td3.select_action(state, noise=exploration_noise)
```
- → `TD3.select_action()` ([td3.py:143](RL/td3.py#L143))
- → **GPU forward pass через actor** (строки 145-146)
- → `.to(device)` + `actor(state)` + `.cpu()` = **~0.3ms**

#### 2. **Critic inference для проверки сертификата** (строка 184)
```python
certified = self.check_lyapunov_certificate(state, action_actor)
```
- → **GPU forward pass через critic** ([calf.py:126-131](RL/calf.py#L126-L131))
- → `.to(device)` + `critic(state, action)` + `.item()` = **~0.5ms**

#### 3. **Повторное вычисление Q-значения** (строка 188, если сертифицирован)
```python
self.update_certificate(state, action_actor)
```
- → **ЕЩЁ ОДИН GPU forward pass через critic** ([calf.py:153-158](RL/calf.py#L153-L158))
- → Повторное вычисление **того же Q-значения** для той же пары (state, action)
- → **~0.5ms**

**Итого на 1 агента**: ~1.3ms (при сертификации)
**На 25 агентов**: **25 × 1.3ms = 32.5ms**
**FPS**: ~30 FPS (только на визуальные агенты!)

---

### **Проблема 3: Избыточные вычисления Q-значений**

**Дублирование работы**:

1. `check_lyapunov_certificate()` (строка 130): вычисляет `q_current`
2. `update_certificate()` (строка 157): **повторно вычисляет тот же `q_value`** для той же пары (state, action)

**Это крайне неэффективно!** Q-значение должно кэшироваться между вызовами.

```python
# check_lyapunov_certificate (строка 130)
q_current, _ = self.td3.critic(state_tensor, action_tensor)  # GPU call #1

# update_certificate (строка 157) - ДЛЯ ТОЙ ЖЕ ПАРЫ (state, action)!
q_value, _ = self.td3.critic(state_tensor, action_tensor)   # GPU call #2 (ДУБЛИРОВАНИЕ!)
```

---

## 📊 Сравнение TD3 vs CALF

### **TD3 версия** ([train_td3_visual.py:452](ursina/train_td3_visual.py#L452))

```python
vis_actions = td3_agent.select_action_batch(vis_states, noise=0.0)
```

**GPU вызовы**:
- ✅ **1 batch forward pass** через actor для 25 агентов
- ✅ Параллельная обработка на GPU (эффективный батчинг)
- ⏱️ **~0.5ms** общее время
- 🚀 **FPS**: 60-120+ (ограничено другими факторами)

**Код TD3.select_action_batch** ([td3.py:154-178](RL/td3.py#L154-L178)):
```python
def select_action_batch(self, states, noise=0.0):
    states_tensor = torch.FloatTensor(states).to(self.device)
    with torch.no_grad():
        actions = self.actor(states_tensor).cpu().data.numpy()  # ОДИН batch forward pass
    # ... add noise if needed ...
    return actions
```

---

### **CALF версия (если цикл)**

```python
for state in vis_states:  # 25 iterations
    action = calf_agent.select_action(state)
```

**GPU вызовы**:
- ❌ **25 sequential forward passes** через actor
- ❌ **25 sequential forward passes** через critic (check_lyapunov)
- ❌ **~25 sequential forward passes** через critic (update_certificate, если сертифицированы)
- ❌ CPU↔GPU синхронизация на каждом вызове `.to(device)` и `.cpu()`
- ⏱️ **~32.5ms** общее время
- 🐌 **FPS**: ~30 (только на визуальные агенты)

**Разница**: **65x медленнее!**

---

## 🎯 Что нужно реализовать в `select_action_batch`

### **Обязательные требования**

#### 1. **Сигнатура метода**
```python
def select_action_batch(self, states, exploration_noise=0.0, return_modes=False):
    """
    Batch version of select_action for efficient multi-agent processing

    Parameters:
    -----------
    states : np.ndarray
        Batch of states, shape (batch_size, state_dim)
    exploration_noise : float
        Exploration noise std (default: 0.0)
    return_modes : bool
        If True, return (actions, modes) where modes is list of action sources

    Returns:
    --------
    actions : np.ndarray
        Batch of actions, shape (batch_size, action_dim)
    modes : list[str] (optional)
        List of action sources: 'td3' (certified), 'relax' (uncertified but relaxed), 'fallback' (nominal policy)
    """
```

---

#### 2. **Batch операции (критично для производительности)**

##### **a) Batch actor inference** (1 вызов вместо 25)
```python
# Используем существующий метод TD3
actions_actor = self.td3.select_action_batch(states, noise=exploration_noise)
```
**Экономия**: 25 вызовов → 1 batch вызов = **~7-8ms**

---

##### **b) Batch critic inference для проверки сертификата** (1 вызов вместо 25)
```python
# Batch forward pass через critic для всех агентов
states_tensor = torch.FloatTensor(states).to(self.device)
actions_tensor = torch.FloatTensor(actions_actor).to(self.device)

with torch.no_grad():
    q_values, _ = self.td3.critic(states_tensor, actions_tensor)
    q_values = q_values.cpu().numpy().flatten()
```
**Экономия**: 25 вызовов → 1 batch вызов = **~12ms**

---

##### **c) Векторизованная проверка сертификатов** (без цикла где возможно)
```python
batch_size = len(states)
certified = np.zeros(batch_size, dtype=bool)

# Если нет сертифицированной тройки, все проходят
if self.q_cert is None:
    certified[:] = True
else:
    # Condition 1: Lyapunov decrease (векторизовано)
    lyapunov_ok = (q_values - self.q_cert) >= self.nu_bar

    # Condition 2: K_infinity bounds (требует вычисления норм)
    state_norms = np.linalg.norm(states, axis=1)  # shape: (batch_size,)
    k_low = self.kappa_low(state_norms)           # shape: (batch_size,)
    k_up = self.kappa_up(state_norms)             # shape: (batch_size,)

    # Векторизованная проверка
    k_infinity_ok = (k_low <= -q_values) & (-q_values <= k_up)

    # Комбинированная проверка
    certified = lyapunov_ok & k_infinity_ok
```

---

#### 3. **Кэширование Q-значений** (избежать повторных forward passes)

**Проблема**: `update_certificate` повторно вычисляет Q-значение

**Решение**: Передавать pre-computed Q-value

##### **Модифицировать `update_certificate`**:
```python
def update_certificate(self, state, action, q_value=None):
    """
    Обновить сертифицированную тройку (s†, a†, q†)

    Parameters:
    -----------
    state : np.ndarray
        State vector
    action : np.ndarray
        Action vector
    q_value : float, optional
        Pre-computed Q-value (to avoid redundant forward pass)
    """
    if q_value is None:
        # Fallback: compute Q-value (для одиночных вызовов select_action)
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action_tensor = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            q_value_tensor, _ = self.td3.critic(state_tensor, action_tensor)
            q_value = q_value_tensor.item()

    self.s_cert = state.copy()
    self.a_cert = action.copy()
    self.q_cert = q_value
    self.q_cert_history.append(q_value)
```

##### **Использовать в `select_action_batch`**:
```python
for i in range(batch_size):
    if certified[i]:
        # Передаём pre-computed Q-value
        self.update_certificate(states[i], actions_actor[i], q_value=q_values[i])
        # Избегаем повторного forward pass!
```

**Экономия**: до 25 forward passes = **~12ms**

---

#### 4. **Определение режимов** (для визуализации трейлов)

```python
modes = []
final_actions = []

for i in range(batch_size):
    self.total_steps += 1

    if certified[i]:
        # Certified: use actor action
        final_actions.append(actions_actor[i])
        modes.append('td3')
        self.update_certificate(states[i], actions_actor[i], q_value=q_values[i])
        self.action_sources.append('td3')
    else:
        # Not certified: relax or fallback
        q = np.random.uniform(0, 1)

        if q >= self.P_relax:
            # Fallback to nominal policy
            action = self.nominal_policy(states[i])
            final_actions.append(action)
            modes.append('fallback')
            self.nominal_interventions += 1
            self.action_sources.append('nominal')
        else:
            # Relax: use actor action anyway
            final_actions.append(actions_actor[i])
            modes.append('relax')
            self.relax_events += 1
            self.action_sources.append('relax')

    # Update P_relax
    self.P_relax *= self.lambda_relax

final_actions = np.array(final_actions)

if return_modes:
    return final_actions, modes
return final_actions
```

---

### **Полный пример реализации**

```python
def select_action_batch(self, states, exploration_noise=0.0, return_modes=False):
    """
    Batch version of select_action for efficient multi-agent processing

    Key optimizations:
    1. Batch actor inference (1 GPU call instead of N)
    2. Batch critic inference (1 GPU call instead of N)
    3. Cached Q-values for update_certificate (avoid N redundant GPU calls)

    Parameters:
    -----------
    states : np.ndarray
        Batch of states, shape (batch_size, state_dim)
    exploration_noise : float
        Exploration noise std
    return_modes : bool
        If True, return (actions, modes)

    Returns:
    --------
    actions : np.ndarray
        Batch of actions, shape (batch_size, action_dim)
    modes : list[str] (optional)
        Action sources for each agent
    """
    batch_size = len(states)

    # Initialize certificate if needed
    if self.s_cert is None:
        # Use first state for initialization
        self.s_cert = states[0].copy()
        self.a_cert = self.nominal_policy(states[0])

        # Compute initial Q-value
        state_tensor = torch.FloatTensor(states[0].reshape(1, -1)).to(self.device)
        action_tensor = torch.FloatTensor(self.a_cert.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            q_value, _ = self.td3.critic(state_tensor, action_tensor)
            self.q_cert = q_value.item()
        self.q_cert_history.append(self.q_cert)

    # OPTIMIZATION 1: Batch actor inference (1 call instead of N)
    actions_actor = self.td3.select_action_batch(states, noise=exploration_noise)

    # OPTIMIZATION 2: Batch critic inference (1 call instead of N)
    states_tensor = torch.FloatTensor(states).to(self.device)
    actions_tensor = torch.FloatTensor(actions_actor).to(self.device)

    with torch.no_grad():
        q_values, _ = self.td3.critic(states_tensor, actions_tensor)
        q_values = q_values.cpu().numpy().flatten()

    # Vectorized certificate checking
    certified = np.ones(batch_size, dtype=bool)

    if self.q_cert is not None:
        # Condition 1: Lyapunov decrease
        lyapunov_ok = (q_values - self.q_cert) >= self.nu_bar

        # Condition 2: K_infinity bounds
        state_norms = np.linalg.norm(states, axis=1)
        k_low = self.kappa_low(state_norms)
        k_up = self.kappa_up(state_norms)
        k_infinity_ok = (k_low <= -q_values) & (-q_values <= k_up)

        certified = lyapunov_ok & k_infinity_ok

    # Process each action
    final_actions = []
    modes = [] if return_modes else None

    for i in range(batch_size):
        self.total_steps += 1

        if certified[i]:
            # OPTIMIZATION 3: Use cached Q-value (avoid redundant forward pass)
            self.update_certificate(states[i], actions_actor[i], q_value=q_values[i])
            final_actions.append(actions_actor[i])
            if return_modes:
                modes.append('td3')
                self.action_sources.append('td3')
        else:
            # Not certified
            q = np.random.uniform(0, 1)

            if q >= self.P_relax:
                # Fallback to nominal policy
                action = self.nominal_policy(states[i])
                final_actions.append(action)
                if return_modes:
                    modes.append('fallback')
                    self.action_sources.append('nominal')
                self.nominal_interventions += 1
            else:
                # Relax
                final_actions.append(actions_actor[i])
                if return_modes:
                    modes.append('relax')
                    self.action_sources.append('relax')
                self.relax_events += 1

        # Update P_relax
        self.P_relax *= self.lambda_relax

    final_actions = np.array(final_actions)

    if return_modes:
        return final_actions, modes
    return final_actions
```

---

## 🎯 Ожидаемый прирост производительности

### **До оптимизации** (если цикл)
| Операция | Количество GPU calls | Время |
|----------|---------------------|-------|
| Actor inference (25×) | 25 | ~7.5ms |
| Critic check (25×) | 25 | ~12.5ms |
| Critic update (25×) | ~25 | ~12.5ms |
| **Итого** | **~75** | **~32.5ms** |
| **FPS** | | **~30** |

### **После оптимизации** (batch)
| Операция | Количество GPU calls | Время |
|----------|---------------------|-------|
| Actor batch inference | 1 | ~0.5ms |
| Critic batch inference | 1 | ~0.5ms |
| Critic update (cached) | 0 | ~0ms |
| **Итого** | **2** | **~1ms** |
| **FPS** | | **500-1000** |

**Реальный FPS** будет ограничен другими факторами:
- Heatmap updates (~100 steps interval)
- Trail rendering (25 trails × rebuilds)
- Ursina rendering overhead
- Python GIL

**Целевой результат**: **сопоставим с TD3** (~60-120 FPS)

**Ускорение**: **~32x**

---

## 📝 Итоговый чеклист для реализации

### **Обязательные изменения**

- [ ] **Реализовать `select_action_batch()` в `CALFController`**
  - [ ] Сигнатура: `(states, exploration_noise=0.0, return_modes=False)`
  - [ ] Возвращает: `actions` или `(actions, modes)`

- [ ] **Batch actor inference**
  - [ ] Использовать `self.td3.select_action_batch(states, noise=...)`
  - [ ] Один вызов для всех агентов

- [ ] **Batch critic inference**
  - [ ] Batch forward pass через critic для всех state-action пар
  - [ ] Сохранить Q-значения для кэширования

- [ ] **Векторизованная проверка сертификатов**
  - [ ] Lyapunov decrease check (векторизовано)
  - [ ] K_infinity bounds check (векторизовано)

- [ ] **Кэширование Q-значений**
  - [ ] Модифицировать `update_certificate()` для приёма `q_value=None`
  - [ ] Передавать pre-computed Q-values в batch режиме
  - [ ] Избегать повторных forward passes

- [ ] **Обработка nominal_policy fallback**
  - [ ] Вызывать `self.nominal_policy(state)` для несертифицированных агентов
  - [ ] Корректно обрабатывать relax events

- [ ] **Возвращать список режимов**
  - [ ] `modes = ['td3', 'relax', 'fallback']` для каждого агента
  - [ ] Только если `return_modes=True`

### **Тестирование**

- [ ] Запустить `train_calf_visual.py`
- [ ] Проверить отсутствие AttributeError
- [ ] Сравнить FPS с `train_td3_visual.py`
- [ ] Проверить корректность цветов трейлов (режимы)
- [ ] Проверить GPU utilization (должна вырасти)

---

## 🔍 Дополнительные гипотетические проблемы

### **Если код всё же работает (маловероятно)**

1. **Monkey patching**: Где-то может быть динамическое добавление метода
   ```python
   # Поиск в коде
   grep -r "select_action_batch.*=" RL/ ursina/
   ```

2. **Fallback в другом месте**: Может быть обработка в родительском классе
   ```python
   # Проверить наследование
   class CALFController:  # Нет родительского класса
   ```

3. **Версия кода отличается**: Возможно, есть другая версия файла
   ```bash
   # Проверить другие версии
   find . -name "calf.py" -type f

   # Проверить git историю
   git log --oneline -- RL/calf.py
   git diff HEAD -- RL/calf.py
   ```

4. **Try-except обработка**: Проверить наличие error handling
   ```bash
   grep -n "try\|except\|AttributeError" ursina/train_calf_visual.py
   # Результат: ничего не найдено
   ```

---

## 🎓 Технические детали

### **Почему нагрузка переместилась с GPU на CPU**

1. **Множество мелких GPU операций**
   - Каждый `.to(device)` требует CPU→GPU transfer
   - Каждый `.cpu()` требует GPU→CPU transfer
   - 75+ transfers/frame вместо 2-3

2. **CPU ждёт GPU**
   - Sequential forward passes не позволяют GPU работать параллельно
   - CPU блокируется на каждом `.item()` (синхронизация)

3. **Memory transfer overhead**
   - 50+ мелких копирований вместо 1-2 больших batch операций
   - Batch operations лучше утилизируют GPU memory bandwidth

4. **Python GIL**
   - Цикл в Python медленнее, чем batch операции в PyTorch C++ backend
   - Vectorized operations обходят GIL

---

## 📚 Ссылки на код

- **Проблемный вызов**: [train_calf_visual.py:402](ursina/train_calf_visual.py#L402)
- **CALFController**: [RL/calf.py](RL/calf.py)
- **select_action**: [RL/calf.py:165](RL/calf.py#L165)
- **check_lyapunov_certificate**: [RL/calf.py:113](RL/calf.py#L113)
- **update_certificate**: [RL/calf.py:151](RL/calf.py#L151)
- **TD3.select_action_batch**: [RL/td3.py:154](RL/td3.py#L154)
- **Сравнение с TD3**: [train_td3_visual.py:452](ursina/train_td3_visual.py#L452)

---

## 🏁 Заключение

**Основная проблема**: Метод `select_action_batch` не реализован в `CALFController`

**Последствия**:
1. AttributeError при запуске (после 1000 шагов)
2. Если есть workaround (цикл) - падение производительности в 32-65x
3. Избыточные GPU вызовы и дублирование вычислений Q-значений

**Решение**: Реализовать batch-оптимизированный `select_action_batch` с кэшированием Q-значений

**Ожидаемый результат**: FPS сопоставим с TD3 (~60-120 FPS), GPU utilization восстановится
