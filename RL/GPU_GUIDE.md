# Руководство по использованию GPU

## Автоматическое определение устройства

Код автоматически определяет наличие GPU и использует его:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## При запуске обучения

Вы увидите информацию об используемом устройстве:

### С GPU:
```
============================================================
Device: cuda
GPU: NVIDIA GeForce RTX 4090
CUDA Version: 12.1
GPU Memory: 24.00 GB
============================================================

TD3 using device: cuda
```

### Без GPU:
```
============================================================
Device: cpu
============================================================

TD3 using device: cpu
```

## Ускорение обучения

### CPU vs GPU
Для этой задачи (маленькие сети, простая среда):
- **CPU**: ~10-15 минут на 300 эпизодов
- **GPU**: ~5-8 минут на 300 эпизодов

**Примечание**: На маленьких сетях (64 hidden units) разница не драматическая, так как большая часть времени уходит на симуляцию среды, а не на обучение нейросетей.

Для более сложных задач с большими сетями GPU даст значительное ускорение.

## Мониторинг GPU

### Использование памяти GPU

Во время обучения можно мониторить использование GPU:

**Windows (NVIDIA):**
```bash
nvidia-smi
```

**Python (во время обучения):**
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Типичное использование памяти

Для данной реализации:
- **Сети**: ~2-5 MB (очень маленькие)
- **Батчи**: ~1-2 MB
- **Общее**: ~10-50 MB (очень мало)

Большая часть памяти GPU будет свободна, так как задача простая.

## Оптимизация для GPU

Если хотите использовать GPU более эффективно:

### 1. Увеличить размер сетей
```python
calf = CALFController(
    hidden_dim=256,  # Вместо 64
    ...
)
```

### 2. Увеличить размер батча
```python
train_calf(
    batch_size=256,  # Вместо 64
    ...
)
```

### 3. Увеличить replay buffer
```python
replay_buffer = ReplayBuffer(
    max_size=1000000,  # Вместо 100000
    ...
)
```

## Troubleshooting

### Ошибка: "CUDA out of memory"

Если видите эту ошибку (маловероятно для данной задачи):

1. Уменьшите batch_size:
```python
batch_size=32  # Вместо 64
```

2. Уменьшите hidden_dim:
```python
hidden_dim=32  # Вместо 64
```

3. Проверьте другие процессы, использующие GPU:
```bash
nvidia-smi
```

### PyTorch не видит GPU

Проверьте установку CUDA:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")
```

Если CUDA недоступна:
1. Установите правильную версию PyTorch для вашей CUDA
2. Проверьте драйверы NVIDIA
3. Перезагрузите систему

## Принудительное использование CPU

Если хотите использовать CPU даже при наличии GPU:

```python
device = torch.device("cpu")

calf = CALFController(
    device=device,
    ...
)
```

## Где используется GPU в коде

### TD3:
1. **Actor/Critic networks**: Размещены на GPU
2. **Forward pass**: Все вычисления на GPU
3. **Backward pass**: Градиенты на GPU
4. **Optimizer**: Обновления на GPU

### CALF:
1. **Lyapunov certificate check**: Q-функция на GPU
2. **Action selection**: Actor на GPU
3. **Training**: TD3 train на GPU

### Evaluation:
1. **Q-heatmap generation**: Все вычисления Q-функции на GPU
2. **Lyapunov heatmap**: -Q вычисления на GPU

## Performance tips

### 1. Mixed Precision Training
Для еще большего ускорения (требует современный GPU):
```python
from torch.cuda.amp import autocast, GradScaler

# В TD3.train():
scaler = GradScaler()

with autocast():
    # Forward pass
    ...

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. Data on GPU
Держать replay buffer на GPU (если хватает памяти):
```python
# В ReplayBuffer.sample():
return (
    torch.FloatTensor(self.states[ind]).to(device),
    ...
)
```

### 3. Асинхронная загрузка
```python
# В TD3.train():
state = state.to(device, non_blocking=True)
action = action.to(device, non_blocking=True)
```

Но для данной задачи это избыточно - она и так быстро обучается.

## Рекомендации

Для данной реализации CALF:
- ✅ GPU полезен, но не критичен
- ✅ Автоматическое определение устройства работает хорошо
- ✅ Не нужно ничего настраивать вручную
- ✅ Код работает одинаково на CPU и GPU

Для более сложных задач:
- Увеличьте размеры сетей
- Используйте более сложные среды
- Тогда GPU даст значительное ускорение (5-10x)
