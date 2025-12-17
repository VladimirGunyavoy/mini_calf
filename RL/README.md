# CALF (Critic as Lyapunov Function) Implementation

Упрощенная реализация CALF с TD3 для задачи стабилизации точки.

## Что это?

Реализация подхода **Critic as Lyapunov Function (CALF)** из диссертации Pavel Osinenko "Reinforcement Learning with Guarantees" (2024).

### Ключевые идеи:

1. **Критик как функция Ляпунова**: -q(s,a) используется как кандидат в функцию Ляпунова
2. **Сертификация действий**: проверка Ляпунов-условий перед выполнением действия
3. **Relax probability**: механизм постепенного уменьшения вероятности "расслабления" защиты
4. **Номинальная политика π₀**: безопасная политика (PD-контроллер), которая гарантирует достижение цели

## Структура проекта

```
CALF/
├── td3.py              # TD3 (Twin Delayed DDPG) алгоритм
├── calf.py             # CALF контроллер с relax probability
├── simple_env.py       # Простая среда: точка в 2D
├── visualize.py        # Визуализация результатов
├── evaluation.py       # Система evaluation во время обучения
├── train_calf.py       # Скрипт обучения
├── calf.md             # Теоретическое описание CALF
└── trainings/          # Результаты обучений (создается автоматически)
    └── run_XXX/        # Папка конкретного запуска
        ├── eval_000_epXXXX/   # Evaluation #0 на эпизоде XXXX
        ├── eval_001_epXXXX/   # Evaluation #1
        ├── ...
        └── last_eval/         # Последний evaluation (обновляется)
```

## Установка зависимостей

```bash
pip install numpy torch matplotlib tqdm
```

### Поддержка GPU

Код автоматически определяет наличие CUDA-совместимого GPU и использует его для ускорения обучения:
- Если GPU доступен → используется `cuda`
- Если нет → используется `cpu`

Для установки PyTorch с поддержкой CUDA:
```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

При запуске отобразится информация об используемом устройстве:
```
============================================================
Device: cuda
GPU: NVIDIA GeForce RTX 4090
CUDA Version: 12.1
GPU Memory: 24.00 GB
============================================================
```

## Запуск

### Обучение CALF:

```bash
python train_calf.py
```

Это запустит обучение на 300 эпизодах с evaluation каждые 20 эпизодов.

**Что создается:**
- `trainings/run_XXX/` - папка текущего запуска
  - `eval_000_ep0020/` - evaluation после 20 эпизодов
    - `training_progress.png` - графики обучения
    - `q_function_trajectories.png` - Q-функция + траектории из буфера
    - `lyapunov_trajectories.png` - функция Ляпунова + траектории
    - `stats.txt` - статистика обучения
  - `eval_001_ep0040/` - evaluation после 40 эпизодов
  - ... (каждые 20 эпизодов)
  - `last_eval/` - **КОПИЯ последнего evaluation** (обновляется каждый раз)
  - `calf_model.pth` - финальная модель

**Особенности:**
- Каждые 20 эпизодов создается snapshot с визуализациями
- Траектории накапливаются между evaluations и отображаются на тепловой карте
- `last_eval/` всегда содержит самые свежие результаты

### Тестирование среды:

```bash
python simple_env.py
```

## Визуализации

### 1. Q-функция с траекториями
Тепловая карта показывает значения Q-функции для разных состояний. Траектории агента показывают, как он движется к цели.

### 2. Функция Ляпунова (-Q)
Показывает -Q(s,a), которая должна убывать к нулю при приближении к цели.

### 3. Фазовый портрет
Траектории в пространстве [позиция, скорость].

### 4. Графики обучения
- Episode Rewards
- Episode Lengths
- Final Distance to Goal
- Intervention Rate (как часто вызывается π₀)
- Relax Rate (как часто происходит "расслабление")

## Параметры CALF

В `train_calf.py`:

- `lambda_relax` - relaxation factor (по умолчанию 0.99)
  - Чем больше, тем дольше агент может "расслабляться"
  - P_relax = λ_relax^t → 0

- `nu_bar` - порог убывания Ляпунова (по умолчанию 0.01)
  - q(s,a) - q†(s†,a†) ≥ ν̄

- `kappa_low_coef`, `kappa_up_coef` - коэффициенты K∞ функций
  - κ_low(r) = C_low * r²
  - κ_up(r) = C_up * r²

## Алгоритм CALF (кратко)

1. Получить действие от актора a ~ π_t(·|s)
2. Проверить Ляпунов-сертификат:
   - q(s,a) - q†(s†,a†) ≥ ν̄
   - κ_low(|s|) ≤ -q(s,a) ≤ κ_up(|s|)
3. Если сертификат прошел → использовать a
4. Если нет:
   - С вероятностью (1 - P_relax) → использовать π₀
   - С вероятностью P_relax → использовать a (расслабиться)
5. Обновить P_relax *= λ_relax

## Результаты

После обучения:
- Агент учится стабилизировать точку в нуле
- P_relax → 0 (защита усиливается со временем)
- Intervention Rate → 0 (реже нужна номинальная политика)
- Агент достигает цели с гарантиями безопасности

## Ссылки

- Pavel Osinenko (2024) "Reinforcement Learning with Guarantees"
- "Critic as Lyapunov function (CALF): a model-free, stability-ensuring agent"
- TD3: "Addressing Function Approximation Error in Actor-Critic Methods"
