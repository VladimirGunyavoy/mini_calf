# Stage 3: Differential Drive - Стартовый промпт

## Точка входа для нового агента

---

## Статус: ✅ ЗАВЕРШЁН

**Проект**: CALF (Critic as Lyapunov Function)
**Задача**: Добавить дифференциальный привод как вторую систему
**Предыдущая стадия**: Stage 2 - Оптимизация (завершена)

---

## Контекст

CALF - это проект обучения с подкреплением с гарантиями безопасности. Критик используется как функция Ляпунова для сертификации действий.

Сейчас система работает с **Point Mass** (точка с массой):
- State: [x, v] - позиция и скорость
- Action: [a] - ускорение

Нужно добавить **Differential Drive** (дифференциальный привод):
- State: [x, y, θ] - позиция и ориентация
- Action: [v, ω] - линейная и угловая скорость

---

## Критические правила

### 1. НЕ ЛОМАЙ POINT MASS
После каждого изменения проверяй что Point Mass ещё работает:
```bash
cd c:\GitHub\Learn\CALF\ursina
py -3.12 main.py
```

### 2. ЖДИ ПОДТВЕРЖДЕНИЯ
Не переходи к следующей части без подтверждения пользователя.

### 3. ЗАПИСЫВАЙ УРОКИ
При решении проблемы - добавь урок в `02_LESSONS_LEARNED.md`.

### 4. ДОКУМЕНТИРУЙ РЕШЕНИЯ
При выборе между вариантами - добавь решение в `03_DECISIONS_LOG.md`.

---

## Где искать информацию

| Что нужно | Где смотреть |
|-----------|--------------|
| План и ТЗ | `01_PLAN.md` |
| Текущая RL среда | `../../RL/simple_env.py` |
| Текущая физика | `../../physics/point_system.py` |
| CALF контроллер | `../../RL/calf.py` |
| Config система | `../../config/training_config.py` |
| Уроки из Stage 2 | `../stage_2/02_LESSONS_LEARNED.md` |

---

## Команды для запуска

```bash
# Перейти в папку
cd c:\GitHub\Learn\CALF\ursina

# Запустить текущую версию (Point Mass)
py -3.12 main.py

# После реализации - запустить с Differential Drive
py -3.12 main.py --system differential_drive

# Тест RL среды
py -3.12 ../RL/differential_drive_env.py
```

---

## Чеклист выполнения

### Part 1: Базовые абстракции ✅
- [x] 1.1 Создать `RL/base_env.py`
- [x] 1.2 Рефакторинг `RL/simple_env.py` (наследование от BaseEnv)
- [x] 1.3 Тест: Point Mass работает

### Part 2: DifferentialDriveEnv ✅
- [x] 2.1 Создать `RL/differential_drive_env.py`
- [x] 2.2 Реализовать динамику, reset, step
- [x] 2.3 Реализовать reward функцию

### Part 3: Номинальная политика ✅
- [x] 3.1 Реализовать `move_to_point_policy()`
- [x] 3.2 Тест: агент достигает [0, 0, 0]

### Part 4: DifferentialDriveSystem ✅
- [x] 4.1 Создать `physics/base_system.py`
- [x] 4.2 Рефакторинг `physics/point_system.py`
- [x] 4.3 Создать `physics/differential_drive_system.py`

### Part 5: Визуализация (Cone) ✅
- [x] 5.1 Создать `visuals/oriented_agent.py`
- [x] 5.2 Тест: cone отображается и вращается

### Part 6: Интеграция (VectorizedEnv, Config) ✅
- [x] 6.1 Добавить `system_type` в VectorizedEnv
- [x] 6.2 Добавить параметры в TrainingConfig
- [x] 6.3 Создать `RL/__init__.py`

### Part 7: Интеграция (main.py, Heatmap) ✅
- [x] 7.1 Добавить argparse `--system` в main.py
- [x] 7.2 Условный выбор среды и политики
- [x] 7.3 Обновить CriticHeatmap (state_dim=3, fixed_theta=0)

### Part 8: Тестирование ✅
- [x] 8.1 `py -3.12 main.py` работает (point_mass)
- [x] 8.2 `py -3.12 main.py --system differential_drive` работает
- [ ] 8.3 Обучение differential_drive сходится (требует длительного теста)

---

## Приоритет

**Сначала RL** (Part 1-3), потом визуализация (Part 4-5), потом интеграция (Part 6-8).

---

## Математическая модель

```
State:  s = [x, y, θ]
Action: a = [v, ω]

Dynamics:
  ẋ = v · cos(θ)
  ẏ = v · sin(θ)
  θ̇ = ω

Reward:
  r = -position_dist - 0.1 * angle_dist - 0.01 * (v² + ω²)

Goal:
  position_dist < 0.1 AND |θ| < 0.1
```

---

## Следующий шаг

1. Открой `01_PLAN.md`
2. Найди Part 1 (Базовые абстракции)
3. Читай ТЗ и начинай реализацию
4. Отмечай галочки в этом файле

---

**Начинай с Part 1!**
