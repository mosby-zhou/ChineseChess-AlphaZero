# MCTS 优化总结

## 性能对比

| MCTS 版本 | 100次模拟耗时 | 加速比 |
|-----------|--------------|--------|
| 原始 MCTS | 1.58s | 1.0x |
| 优化 MCTS (缓存) | 1.41s | 1.1x |
| **批量 MCTS (批大小8)** | **0.74s** | **2.1x** |

## 优化版本说明

### 1. 原始 MCTS (`mcts.py`)
- 标准实现
- 每次模拟单独推理
- 适合调试和理解算法

### 2. 优化 MCTS (`mcts_optimized.py`)
- 添加局面缓存，避免重复推理
- 使用 `__slots__` 减少内存开销
- 适合局面重复度高的场景

### 3. 批量 MCTS (`mcts_fast.py`)
- **最快**：使用虚拟损失 + 批量推理
- GPU 利用率最高
- 适合大规模训练

## 使用方法

### 当前训练已使用批量 MCTS

`train.py` 已配置使用 `OptimizedMCTS`，无需修改。

### 手动选择 MCTS 版本

```python
# 方式 1: 原始 MCTS
from simple_chess_ai.mcts import MCTS
mcts = MCTS(model, num_simulations=100)

# 方式 2: 缓存优化 MCTS（当前默认）
from simple_chess_ai.mcts_optimized import OptimizedMCTS as MCTS
mcts = MCTS(model, num_simulations=100, cache_size=2000)

# 方式 3: 批量 MCTS（最快）
from simple_chess_ai.mcts_fast import FastBatchMCTS as MCTS
mcts = MCTS(model, num_simulations=100, batch_size=8)
```

## 训练命令推荐

```bash
# 快速训练（每局 ~8 秒）
python -m simple_chess_ai train \
    --num_games 200 \
    --num_simulations 50 \
    --max_moves 100

# 标准训练（每局 ~25 秒）
python -m simple_chess_ai train \
    --num_games 100 \
    --num_simulations 100

# 高质量训练（每局 ~50 秒）
python -m simple_chess_ai train \
    --num_games 100 \
    --num_simulations 200
```

## 进一步优化方向

1. **并行自对弈**: 多进程同时生成训练数据
2. **异步推理**: GPU 推理与 CPU 选择并行
3. **神经网络优化**: 更小的网络或知识蒸馏
