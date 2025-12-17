# Руководство по Evaluation System

## Как это работает

Система evaluation автоматически создает визуализации и сохраняет прогресс обучения каждые N эпизодов (по умолчанию 20).

## Структура папок

```
trainings/
└── run_000/                          # Запуск #0
    ├── eval_000_ep0020/              # Evaluation после 20 эпизодов
    │   ├── training_progress.png     # 6 графиков обучения
    │   ├── q_function_trajectories.png      # Q-функция + траектории
    │   ├── lyapunov_trajectories.png        # Ляпунов + траектории
    │   ├── model.pth                 # Веса TD3 (actor, critic)
    │   ├── model_calf.npz            # Параметры CALF (сертификат, P_relax)
    │   └── stats.txt                 # Статистика
    ├── eval_001_ep0040/              # Evaluation после 40 эпизодов
    │   └── ...
    ├── eval_XXX_epYYYY/              # Evaluation #XXX на эпизоде YYYY
    │   └── ...
    ├── last_eval/                    # Копия последнего evaluation
    │   ├── training_progress.png     # ← Обновляется каждый eval
    │   ├── q_function_trajectories.png
    │   ├── lyapunov_trajectories.png
    │   ├── model.pth                 # ← Последние веса
    │   ├── model_calf.npz            # ← Последние параметры CALF
    │   └── stats.txt
    └── calf_model.pth                # Финальная модель (последний эпизод)
```

## Что происходит во время evaluation

### 1. Накопление траекторий
Между evaluations все траектории эпизодов сохраняются в буфер:
- После каждого эпизода траектория добавляется в `evaluator.trajectory_buffer`
- При evaluation все траектории из буфера отображаются на тепловой карте
- После evaluation буфер очищается

**Пример:**
- Эпизоды 1-20: накапливаем 20 траекторий
- Evaluation #0: рисуем Q-функцию + эти 20 траекторий
- Буфер очищается
- Эпизоды 21-40: накапливаем следующие 20 траекторий
- Evaluation #1: рисуем Q-функцию + новые 20 траекторий
- ...

### 2. Создаваемые файлы

#### training_progress.png
6 графиков:
1. Episode Rewards (с moving average)
2. Episode Lengths (с moving average)
3. Final Distance to Goal (с moving average, log scale)
4. Nominal Policy Intervention Rate
5. Relax Event Rate
6. Combined Rates (Intervention + Relax)

#### q_function_trajectories.png
- Тепловая карта Q-функции (текущее состояние сети)
- Все траектории из буфера (красные линии)
- Начальные точки (зеленые кружки)
- Конечные точки (красные квадраты)
- Целевая область (красный круг)

#### lyapunov_trajectories.png
- Тепловая карта функции Ляпунова: -Q(s,a)
- Все траектории из буфера (синие линии)
- Начальные/конечные точки
- Целевая область (синий круг)

#### stats.txt
Текстовый файл со статистикой:
- CALF statistics (interventions, relax events, P_relax)
- Training statistics (средние метрики за последние 20 эпизодов)
- Количество траекторий в буфере
- Информация о сохраненной модели

#### model.pth
PyTorch checkpoint с весами нейросетей:
- Actor network (текущая политика)
- Critic networks (Q-функции)
- Actor optimizer state
- Critic optimizer state

#### model_calf.npz
NumPy archive с параметрами CALF:
- s_cert, a_cert, q_cert (сертифицированная тройка)
- P_relax (текущая relax probability)
- Счетчики (total_steps, nominal_interventions, relax_events)

## Папка last_eval

`last_eval/` - это **копия** последнего evaluation.

**Зачем нужна:**
- Быстрый доступ к самым свежим результатам
- Не нужно искать последнюю папку eval_XXX
- Удобно для мониторинга во время обучения

**Как обновляется:**
После каждого evaluation все файлы копируются в `last_eval/`:
```python
shutil.copy2(src, dst)  # Копирование с сохранением метаданных
```

## Использование в коде

### Базовое использование

```python
from evaluation import TrainingEvaluator

# Создать evaluator
evaluator = TrainingEvaluator(save_dir='trainings')

# В цикле обучения
for episode in range(num_episodes):
    # ... обучение ...

    # Добавить траекторию
    evaluator.add_trajectory(trajectory)

    # Evaluation каждые 20 эпизодов
    if (episode + 1) % 20 == 0:
        evaluator.evaluate(
            calf=calf,
            env=env,
            episode=episode + 1,
            total_episodes=num_episodes,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            final_distances=final_distances,
            intervention_rates=intervention_rates,
            relax_rates=relax_rates
        )
```

### Настройка интервала evaluation

```python
# Evaluation каждые 10 эпизодов
calf, env, rewards, lengths, evaluator = train_calf(
    num_episodes=300,
    eval_interval=10,  # ← Изменить интервал
    ...
)
```

### Ручное управление буфером траекторий

```python
# Добавить траекторию вручную
trajectory = np.array([[x1, v1], [x2, v2], ...])
evaluator.add_trajectory(trajectory)

# Очистить буфер
evaluator.clear_trajectory_buffer()

# Получить количество траекторий
num_traj = len(evaluator.trajectory_buffer)
```

## Мониторинг во время обучения

### Вариант 1: Смотреть last_eval
```bash
# В процессе обучения открывайте файлы из last_eval
# Они обновляются каждые 20 эпизодов
ls trainings/run_000/last_eval/
```

### Вариант 2: Прогресс-бар tqdm
```
Training: 45%|████▌     | 135/300 [02:15<02:45,  1.00it/s,
  R=-45.2, Len=234, Dist=0.0823, Goal=True,
  P_relax=0.000123, Interv=0.045, Loss=0.3421]
```

### Вариант 3: Детальный вывод каждые 20 эпизодов
```
====================================
EVALUATION #2 at Episode 40/300
====================================
Plotting training progress...
Plotting Q-function with 20 trajectories...
Plotting Lyapunov function with 20 trajectories...
Copying to last_eval...
Evaluation saved to: trainings/run_000/eval_002_ep0040
Last eval updated: trainings/run_000/last_eval
====================================
```

## Загрузка моделей из evaluation

### Загрузка модели с определенного evaluation

```python
from calf import CALFController
from simple_env import PointMassEnv, pd_nominal_policy

# Создать среду и номинальную политику
env = PointMassEnv()
nominal_policy = pd_nominal_policy()

# Создать CALF контроллер
calf = CALFController(
    state_dim=2,
    action_dim=1,
    max_action=5.0,
    nominal_policy=nominal_policy
)

# Загрузить модель из evaluation #5 (эпизод 100)
model_path = 'trainings/run_000/eval_005_ep0100/model.pth'
calf.load(model_path)

# Теперь можно использовать модель
state = env.reset()
action = calf.select_action(state, exploration_noise=0.0)
```

### Загрузка последней модели

```python
# Загрузить из last_eval (самая свежая модель)
model_path = 'trainings/run_000/last_eval/model.pth'
calf.load(model_path)
```

### Сравнение моделей из разных evaluation

```python
import matplotlib.pyplot as plt

# Загрузить модели из разных этапов
eval_episodes = [20, 100, 200, 300]
models = []

for ep in eval_episodes:
    eval_num = (ep // 20) - 1
    model_path = f'trainings/run_000/eval_{eval_num:03d}_ep{ep:04d}/model.pth'

    calf = CALFController(...)
    calf.load(model_path)
    models.append((ep, calf))

# Тестировать каждую модель
for ep, calf in models:
    # Запустить траектории
    trajectory, _, _ = rollout_trajectory(calf, env)
    plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'Episode {ep}')

plt.legend()
plt.show()
```

## Советы

1. **Регулярно проверяйте last_eval** - там всегда свежие результаты

2. **Настраивайте eval_interval** в зависимости от длительности обучения:
   - Короткое обучение (100 эпизодов) → eval_interval=10
   - Среднее (300 эпизодов) → eval_interval=20
   - Длинное (1000+ эпизодов) → eval_interval=50

3. **Траектории на тепловой карте** показывают:
   - Как агент исследует пространство состояний
   - Сходятся ли траектории к цели
   - Есть ли проблемные области

4. **Функция Ляпунова** должна:
   - Убывать к нулю в центре (целевая область)
   - Расти от центра (чем дальше от цели, тем больше -Q)
   - Траектории должны идти "вниз по склону"

5. **Сравнение разных запусков**:
   ```
   trainings/run_000/  # λ_relax = 0.99
   trainings/run_001/  # λ_relax = 0.95
   trainings/run_002/  # λ_relax = 0.90
   ```

## Пример: Анализ обучения

1. Откройте `last_eval/training_progress.png`
   - Проверьте сходимость rewards
   - Distance должен убывать
   - Intervention rate должен падать

2. Откройте `last_eval/q_function_trajectories.png`
   - Траектории должны сходиться к центру
   - Q-функция должна быть максимальной в центре

3. Откройте `last_eval/stats.txt`
   - Проверьте P_relax (должен стремиться к 0)
   - Intervention rate (должен уменьшаться)
   - Relax rate (должен падать)

4. Если что-то не так:
   - Посмотрите предыдущие evaluations
   - Сравните Q-функции в разные моменты времени
   - Проверьте, как менялись траектории
