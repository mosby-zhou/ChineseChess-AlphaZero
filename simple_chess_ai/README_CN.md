# 新旧项目能力对比评估报告

## 概述

本报告对比分析 `cchess_alphazero`（旧项目）与 `simple_chess_ai`（新项目）的技术架构、模型能力、训练流程，评估新项目是否能对齐和超越旧项目，以及是否符合当前业界最新趋势。

---

## 一、架构对比

### 1.1 技术栈

| 维度         | 旧项目 (cchess_alphazero)    | 新项目 (simple_chess_ai) |
| ------------ | ---------------------------- | ------------------------ |
| 深度学习框架 | TensorFlow 1.3 + Keras 2.0.8 | PyTorch（最新版）        |
| 语言版本     | Python 3.6.3                 | Python 3.8+              |
| GPU 支持     | CUDA（旧版）                 | CUDA + DirectML（AMD）   |

**评估**：新项目采用 PyTorch，具备以下优势：

- 更灵活的动态计算图，便于实验和调试
- 更好的社区支持和生态
- DirectML 支持 AMD GPU
- **结论：新项目技术栈更现代，对齐业界主流**

### 1.2 网络架构

| 维度       | 旧项目           | 新项目                   |
| ---------- | ---------------- | ------------------------ |
| 网络类型   | 纯 CNN（ResNet） | CNN（默认）/ GNN（可选） |
| 通道数     | 256              | 128                      |
| 残差块数量 | 7                | 4                        |
| 参数量     | ~5M              | ~2M                      |

**注意**：GNN 后端在 DirectML 上兼容性不佳，推荐使用 CNN 后端。

---

## 二、业界趋势符合度

| 趋势                      | 旧项目 | 新项目        |
| ------------------------- | ------ | ------------- |
| **图神经网络（GNN）**     | ❌     | ✅（可选）    |
| **混合精度训练（FP16）**  | ❌     | ✅（仅 CUDA） |
| **GRPO / 强化学习新方法** | ❌     | ✅            |
| **分布式训练**            | ✅     | ❌            |
| **模型轻量化**            | ❌     | ✅            |

---

## 三、总结

**新项目能否对齐旧项目？**

- 核心能力：✅ 可以对齐（规则、网络、训练流程）
- 训练规模：❌ 暂时无法对齐（缺少分布式）

**是否符合业界最新趋势？**

- ✅ 完全符合，且部分领先（GNN + GRPO）

---

_报告生成时间：2026-03-18_

---

# 用户使用手册

## 一、环境准备

### 1.1 系统要求

| 配置项 | 最低要求       | 推荐配置   |
| ------ | -------------- | ---------- |
| Python | 3.8+           | 3.10+      |
| 内存   | 4GB            | 8GB+       |
| 存储   | 500MB          | 2GB        |
| GPU    | 无（可用 CPU） | AMD/NVIDIA |

### 1.2 创建虚拟环境

```bash
# 进入项目目录
cd /home/zhouc/code/2026/ChineseChess-AlphaZero

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate
```

### 1.3 安装依赖

**AMD GPU（DirectML）**：

```bash
pip install torch-directml numpy pygame matplotlib
```

**NVIDIA GPU（CUDA）**：

```bash
pip install torch numpy pygame matplotlib --index-url https://download.pytorch.org/whl/cu130
```

**纯 CPU**：

```bash
pip install torch numpy pygame matplotlib --index-url https://download.pytorch.org/whl/cpu
```

### 1.4 验证安装

```bash
# 验证 GPU 是否可用
python -c "
import torch_directml as dml
print(f'GPU: {dml.device_name(0)}')
print(f'数量: {dml.device_count()}')
"
```

预期输出：

```
GPU: AMD Radeon Pro 5300M
数量: 1
```

---

## 二、快速开始

### 2.1 快速验证（约 5 秒）

```bash
python -m simple_chess_ai train --quick
```

预期输出：

```
创建新模型
开始训练 (Standard)
自对弈局数: 1
MCTS模拟次数: 10
[第 1/1 局] 胜方: 和棋, 步数: 50, ...
训练完成！
```

### 2.2 正式训练

```bash
# 基础训练（50局，约 20-30 分钟）
python -m simple_chess_ai train \
    --num_games 50 \
    --num_simulations 100 \
    --batch_size 256

# 长时间训练（200局，约 2 小时）
python -m simple_chess_ai train \
    --num_games 200 \
    --num_simulations 200 \
    --gating_interval 20 \
    --gating_games 20
```

### 2.3 人机对弈

```bash
# 命令行对弈（红方执先）
python -m simple_chess_ai play_cli --human_color red

# 图形界面对弈
python -m simple_chess_ai play
```

**走法格式**：`x0 y0 x1 y1`

- 例如 `4 0 4 1` 表示帅从 (4,0) 向前走一步

---

## 三、完整命令参考

### 3.1 训练命令

```bash
python -m simple_chess_ai train [选项]
```

| 选项                | 默认值                | 说明                      |
| ------------------- | --------------------- | ------------------------- |
| `--num_games`       | 50                    | 自对弈局数                |
| `--num_simulations` | 100                   | 每步 MCTS 模拟次数        |
| `--num_epochs`      | 5                     | 每次训练轮数              |
| `--batch_size`      | 256                   | 训练批大小                |
| `--lr`              | 0.001                 | 学习率                    |
| `--max_moves`       | 200                   | 每局最大步数              |
| `--buffer_size`     | 10000                 | 数据缓冲区大小            |
| `--model_path`      | saved_model/model.pth | 模型保存路径              |
| `--save_interval`   | 10                    | 每隔多少局保存模型        |
| `--use_grpo`        | False                 | 使用 GRPO 训练            |
| `--grpo_group_size` | 8                     | GRPO 组大小               |
| `--gating_interval` | 20                    | Gating 评测间隔（0=禁用） |
| `--gating_games`    | 20                    | Gating 对局数             |
| `--gating_winrate`  | 0.55                  | Gating 接受阈值           |
| `--seed`            | None                  | 随机种子                  |
| `--quick`           | False                 | 快速验证模式              |

### 3.2 对弈命令

```bash
# 命令行对弈
python -m simple_chess_ai play_cli [--num_simulations 200] [--human_color red]

# 图形界面对弈
python -m simple_chess_ai play [--num_simulations 200]
```

---

## 四、AMD GPU 配置指南

### 4.1 已验证环境

| 项目    | 配置                 |
| ------- | -------------------- |
| 系统    | WSL2 + Ubuntu 24.04  |
| GPU     | AMD Radeon Pro 5300M |
| PyTorch | torch-directml 0.2.5 |
| 后端    | CNN（纯卷积）        |

### 4.2 注意事项

1. **后端选择**：使用 CNN 后端（默认），GNN 后端在 DirectML 上兼容性不佳
2. **FP16 不支持**：DirectML 不支持 FP16 混合精度，训练时不要使用 `--use_fp16`
3. **性能**：单局训练约 20-30 秒（100次模拟）

### 4.3 常见警告

```
UserWarning: The operator 'aten::lerp.Scalar_out' is not currently supported on the DML backend...
```

这是正常的，不影响训练，部分 Adam 优化器操作会回退到 CPU。

---

## 五、文件结构

```
simple_chess_ai/
├── saved_model/
│   ├── model.pth          # best 模型
│   ├── last.pth           # 最新候选模型
│   └── config.json        # 训练配置
├── runs/
│   └── run_YYYYMMDD_HHMMSS/
│       ├── self_play.jsonl   # 自对弈记录
│       ├── training.csv      # 训练指标
│       ├── gating.csv        # Gating 结果
│       └── plots/            # 训练曲线图
├── game.py           # 象棋规则
├── model.py          # 策略价值网络
├── mcts.py           # MCTS 搜索
├── train.py          # 训练流程
├── cli.py            # 命令行对弈
└── gui.py            # 图形界面
```

---

## 六、常见问题

### Q1: 训练报错 "Weights only load failed"

**A**: 已修复。如果仍遇到，删除旧模型重新训练：

```bash
rm -rf simple_chess_ai/saved_model/*
python -m simple_chess_ai train --quick
```

### Q2: Gating 报错 "Missing key(s) in state_dict"

**A**: 已修复。确保使用最新的代码。

### Q3: GPU 识别为 CPU

**A**: 检查 DirectML 安装：

```bash
python -c "import torch_directml; print(torch_directml.device_name(0))"
```

如果报错，重新安装：

```bash
pip install --force-reinstall torch-directml
```

### Q4: 训练很慢（单局超过 1 分钟）

**A**: 确认使用了 CNN 后端（默认），不是 GNN。检查输出是否显示警告。

---

## 七、参数详解

### 7.1 核心参数说明

#### `--use_fp16` 混合精度训练

| 项目 | 说明 |
|------|------|
| **作用** | 使用 FP16（半精度浮点）进行训练，减少显存占用，加速计算 |
| **支持平台** | ✅ NVIDIA GPU（CUDA） / ❌ AMD GPU（DirectML） |
| **显存节省** | 约 40-50% |
| **速度提升** | 约 1.5-2x |
| **精度影响** | 极小（现代 FP16 实现已很成熟） |
| **适用场景** | 显存不足、追求训练速度 |

**使用建议**：
- NVIDIA GPU（如 RTX 5070Ti）：**推荐开启**
- AMD GPU：**不可用**，会报错或自动回退 FP32

```bash
# NVIDIA GPU 推荐用法
python -m simple_chess_ai train --use_fp16
python -m simple_chess_ai train \
    --num_games 100 \
    --num_simulations 100 \
    --batch_size 256 \
    --gating_interval 20
```

### NVIDIA GPU（可用 FP16 加速）

● 两个命令的区别：

┌───────────────────┬─────────────────┬────────────┬────────────────────┐  
 │ 参数 │ 命令1 │ 命令2 │ 影响 │
├───────────────────┼─────────────────┼────────────┼────────────────────┤  
 │ --num_games │ 100 │ 200 │ 训练局数 │  
 ├───────────────────┼─────────────────┼────────────┼────────────────────┤  
 │ --num_simulations │ 200 │ 50 │ 每步 MCTS 模拟次数 │  
 ├───────────────────┼─────────────────┼────────────┼────────────────────┤  
 │ --batch_size │ 512 │ 256 (默认) │ 训练批次大小 │  
 ├───────────────────┼─────────────────┼────────────┼────────────────────┤  
 │ --mcts_mode │ standard (默认) │ batch │ MCTS 实现方式 │
├───────────────────┼─────────────────┼────────────┼────────────────────┤  
 │ --gating_interval │ 20 │ 50 (默认) │ 模型评估频率 │
└───────────────────┴─────────────────┴────────────┴────────────────────┘

核心差异：

┌──────────┬────────────────────────┬────────────────────┐  
 │ │ 命令1 (高质量) │ 命令2 (高效率) │
├──────────┼────────────────────────┼────────────────────┤  
 │ 特点 │ 每步思考深 (200次模拟) │ 批量推理加速 │
├──────────┼────────────────────────┼────────────────────┤  
 │ 棋力 │ 更强（搜索更充分） │ 稍弱（搜索少但快） │  
 ├──────────┼────────────────────────┼────────────────────┤  
 │ 速度 │ 每局较慢 │ 每局较快 (~1.8x) │  
 ├──────────┼────────────────────────┼────────────────────┤  
 │ 内存 │ 较高 (batch_size=512) │ 较低 │
├──────────┼────────────────────────┼────────────────────┤  
 │ 适用场景 │ 追求高质量训练 │ 快速迭代实验 │
└──────────┴────────────────────────┴────────────────────┘

建议：

- 追求棋力：命令1（高模拟次数）
- 快速迭代：命令2（batch 模式 + 低模拟次数）
- 最佳组合：--mcts_mode batch --num_simulations 200（兼顾速度和质量）

```bash
python -m simple_chess_ai train \
    --num_games 100 \
    --num_simulations 200 \
    --batch_size 512 \
    --use_fp16 \
    --gating_interval 20
```

```bash
python -m simple_chess_ai train --num_games 200 --mcts_mode batch --num_simulations 50 --use_fp16
```

---

#### `--use_grpo` GRPO 训练方法

| 项目 | 说明 |
|------|------|
| **全称** | Group Relative Policy Optimization（组相对策略优化）|
| **作用** | 一种新型强化学习训练方法，相比传统 AlphaZero 方法更稳定 |
| **原理** | 在一个轨迹组内比较多个采样，选择相对最优的策略更新 |
| **相关参数** | `--grpo_group_size`（默认 8）控制每次采样的轨迹组大小 |

**GRPO vs 传统 AlphaZero**：

| 对比项 | 传统 AlphaZero | GRPO |
|--------|----------------|------|
| 更新方式 | 绝对值监督学习 | 组内相对比较 |
| 稳定性 | 较低（易震荡）| 较高 |
| 收敛速度 | 较慢 | 较快 |
| 显存需求 | 低 | 较高（需存储多条轨迹）|
| 适用场景 | 大规模训练 | 中小规模快速收敛 |

**使用建议**：
- 快速实验、中小规模训练：**推荐开启**
- 大规模分布式训练：传统方法可能更稳定

```bash
python -m simple_chess_ai train --use_grpo --grpo_group_size 8
```

---

#### `--batch_size` 训练批次大小

| 项目 | 说明 |
|------|------|
| **作用** | 每次训练迭代使用的样本数量 |
| **默认值** | 256 |
| **影响** | 更大的 batch_size → 更稳定的梯度，但需要更多显存 |

**显存参考（CNN 后端）**：

| batch_size | FP32 显存 | FP16 显存 |
|------------|-----------|-----------|
| 128 | ~2GB | ~1.2GB |
| 256 | ~3GB | ~1.8GB |
| 512 | ~5GB | ~3GB |
| 1024 | ~9GB | ~5GB |

**使用建议**：
- RTX 5070Ti（16GB 显存）：推荐 `--batch_size 512` 或 `1024`
- RTX 3060（12GB 显存）：推荐 `--batch_size 256` 或 `512`
- RTX 2060（6GB 显存）：推荐 `--batch_size 128`

```bash
# 根据显存调整
python -m simple_chess_ai train --batch_size 512 --use_fp16
```

---

#### `--mcts_mode` MCTS 搜索模式

| 模式 | 说明 | 速度 | 棋力 |
|------|------|------|------|
| `standard` | 标准 MCTS，逐个节点展开 | 基准 | 基准 |
| `optimized` | 优化版，带节点缓存 | ~1.2x | 相同 |
| `batch` | 批量推理，并行展开多个节点 | ~1.8x | 略低 |

**工作原理**：

| 模式 | 推理方式 |
|------|----------|
| `standard` | 每次推理 1 个节点 |
| `optimized` | 每次推理 1 个节点 + LRU 缓存 |
| `batch` | 批量推理 8 个节点（Virtual Loss 并行）|

**使用建议**：
- 追求速度（快速迭代）：`--mcts_mode batch`
- 追求棋力（高质量训练）：`--mcts_mode optimized` 或 `standard`

```bash
# 快速迭代
python -m simple_chess_ai train --mcts_mode batch --num_simulations 50

# 高质量训练
python -m simple_chess_ai train --mcts_mode optimized --num_simulations 200
```

---

#### CNN vs GNN 网络架构

| 项目 | CNN（卷积神经网络）| GNN（图神经网络）|
|------|-------------------|------------------|
| **原理** | 将棋盘视为图像，提取局部特征 | 将棋盘视为图结构，节点是棋子 |
| **优势** | 成熟稳定、速度快、兼容性好 | 更好地建模棋子关系、理论上限更高 |
| **劣势** | 对远距离棋子关系建模较弱 | 计算量大、部分平台不支持 |
| **兼容性** | ✅ CUDA ✅ DirectML ✅ CPU | ✅ CUDA ❌ DirectML ✅ CPU |
| **参数量** | ~2M | ~3M |
| **训练速度** | **快（基准）** | 较慢（约 CNN 的 60-70%）|

**使用建议**：

| 平台 | 推荐架构 |
|------|----------|
| NVIDIA GPU（CUDA）| CNN（稳定）或 GNN（实验性）|
| AMD GPU（DirectML）| **仅 CNN**（GNN 不兼容）|
| CPU | CNN |

**切换方式**：
```bash
# 使用 CNN（默认，速度快）
python -m simple_chess_ai train --backend cnn

# 使用 GNN（理论上限高，速度较慢）
python -m simple_chess_ai train --backend gnn
```

---

#### `--num_simulations` MCTS 模拟次数

| 项目 | 说明 |
|------|------|
| **作用** | 每步棋 MCTS 搜索的模拟次数 |
| **默认值** | 100 |
| **影响** | 模拟次数越多，搜索越充分，棋力越强，但速度越慢 |

**推荐配置**：

| 目标 | 模拟次数 | 单局耗时（参考）|
|------|----------|-----------------|
| 快速验证 | 10-20 | 5-10秒 |
| 快速迭代 | 50 | 15-30秒 |
| 标准训练 | 100 | 30-60秒 |
| 高质量训练 | 200-400 | 1-3分钟 |
| 比赛级 | 800+ | 5分钟+ |

```bash
# 快速验证
python -m simple_chess_ai train --quick  # 自动设为 10

# 高质量训练
python -m simple_chess_ai train --num_simulations 400
```

---

#### `--gating_interval` 模型评估间隔

| 项目 | 说明 |
|------|------|
| **作用** | 每隔多少局进行一次候选模型 vs 最佳模型的对弈评估 |
| **默认值** | 20 |
| **设为 0** | 禁用 Gating 机制，候选模型直接替换最佳模型 |

**Gating 机制说明**：
- 候选模型与当前最佳模型对弈 `--gating_games` 局
- 如果候选模型胜率 ≥ `--gating_winrate`（默认 55%），则替换最佳模型
- 防止模型退化，确保持续进步

**使用建议**：
- 正式训练：`--gating_interval 20`
- 快速实验：`--gating_interval 0`（禁用，直接替换）

---

### 7.2 完整参数表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_games` | 50 | 自对弈局数 |
| `--num_simulations` | 100 | 每步 MCTS 模拟次数 |
| `--num_epochs` | 5 | 每局后的训练轮数 |
| `--batch_size` | 256 | 训练批次大小 |
| `--lr` | 0.001 | 学习率 |
| `--max_moves` | 200 | 每局最大步数（超过判和）|
| `--buffer_size` | 10000 | 数据缓冲区大小 |
| `--model_path` | None | 加载已有模型路径 |
| `--save_interval` | 10 | 每隔多少局保存模型 |
| `--use_grpo` | False | 使用 GRPO 训练方法 |
| `--grpo_group_size` | 8 | GRPO 组大小 |
| `--use_fp16` | False | 使用 FP16 混合精度（仅 CUDA）|
| `--gating_interval` | 20 | Gating 评估间隔（0=禁用）|
| `--gating_games` | 20 | Gating 对弈局数 |
| `--gating_winrate` | 0.55 | Gating 接受阈值 |
| `--mcts_mode` | optimized | MCTS 模式：standard/optimized/batch |
| `--mcts_batch_size` | 8 | batch 模式的批量大小 |
| `--backend` | cnn | 网络架构：cnn（速度快）/ gnn（理论上限高）|
| `--seed` | None | 随机种子 |
| `--deterministic` | False | 确定性模式（可复现）|
| `--quick` | False | 快速验证模式 |

---

## 八、推荐训练配置

### 8.1 NVIDIA GPU（Windows + CUDA + RTX 5070Ti）

**推荐命令**：

```bash
# 高质量训练（推荐）
python -m simple_chess_ai train `
    --num_games 200 `
    --num_simulations 200 `
    --batch_size 512 `
    --use_fp16 `
    --mcts_mode batch `
    --backend cnn `
    --gating_interval 20

# 快速迭代实验
python -m simple_chess_ai train `
    --num_games 100 `
    --num_simulations 100 `
    --batch_size 2048 `
    --use_fp16 `
    --mcts_mode batch `
    --backend cnn `
    --use_grpo

# GNN 实验性训练（理论上限高，速度较慢）
python -m simple_chess_ai train `
    --num_games 100 `
    --num_simulations 100 `
    --batch_size 256 `
    --use_fp16 `
    --backend gnn
```

**配置说明**：

| 参数 | 推荐值 | 原因 |
|------|--------|------|
| `--use_fp16` | ✅ 开启 | RTX 5070Ti 支持 FP16，加速 ~2x |
| `--batch_size` | 512 | 16GB 显存充足 |
| `--mcts_mode` | batch | NVIDIA GPU 推理快，批量效率高 |
| `--backend` | cnn | CNN 速度快，GNN 可实验性尝试 |
| `--num_simulations` | 200 | 平衡质量与速度 |
| `--use_grpo` | 可选 | 快速收敛实验可开启 |

---

### 8.2 AMD GPU / DirectML

```bash
# AMD GPU 推荐（仅支持 CNN）
python -m simple_chess_ai train \
    --num_games 100 \
    --num_simulations 100 \
    --batch_size 256 \
    --mcts_mode optimized \
    --backend cnn \
    --gating_interval 20
```

**注意事项**：
- ❌ 不支持 `--use_fp16`
- ❌ 不支持 GNN 后端（仅 CNN，`--backend gnn` 会报错）
- ✅ 可使用 batch 模式 MCTS

---

### 8.3 纯 CPU

```bash
# CPU 推荐（小批量）
python -m simple_chess_ai train \
    --num_games 20 \
    --num_simulations 50 \
    --batch_size 128
```

---

## 九、训练流程图

```
┌─────────────────────────────────────────────────────────────┐
│                      训练主循环                              │
├─────────────────────────────────────────────────────────────┤
│  1. 自对弈生成数据                                           │
│     └─> MCTS 搜索 → 选择动作 → 执行 → 记录 (state, π, z)     │
│                                                             │
│  2. 数据存入缓冲区                                           │
│     └─> 保留最近 N 局数据 (buffer_size)                      │
│                                                             │
│  3. 神经网络训练                                             │
│     └─> 采样 batch → 计算损失 → 反向传播 → 更新参数          │
│                                                             │
│  4. Gating 评估（可选）                                      │
│     └─> 候选模型 vs 最佳模型对弈 → 胜率 ≥ 55% → 替换        │
│                                                             │
│  5. 循环直到达到目标局数                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 十、常见问题

### Q1: 训练报错 "Weights only load failed"

**A**: 已修复。如果仍遇到，删除旧模型重新训练：
```bash
rm -rf simple_chess_ai/saved_model/*
python -m simple_chess_ai train --quick
```

### Q2: Gating 报错 "Missing key(s) in state_dict"

**A**: 已修复。确保使用最新的代码。

### Q3: GPU 识别为 CPU

**A**: 检查 GPU 驱动和 PyTorch 安装：
```bash
# NVIDIA
python -c "import torch; print(torch.cuda.is_available())"

# AMD
python -c "import torch_directml; print(torch_directml.device_name(0))"
```

### Q4: 训练很慢（单局超过 1 分钟）

**A**:
1. 确认 GPU 被正确识别
2. NVIDIA 用户尝试 `--use_fp16 --mcts_mode batch`
3. 降低 `--num_simulations`

### Q5: CUDA out of memory

**A**: 减小 batch_size：
```bash
python -m simple_chess_ai train --batch_size 256  # 或 128
```

### Q6: FP16 报错或警告

**A**:
- NVIDIA GPU：确保 CUDA 版本 ≥ 11.0
- AMD GPU：不支持 FP16，请移除 `--use_fp16`

---

*使用手册版本：v3.0*
*更新时间：2026-03-22*
*测试环境：WSL2 + AMD Radeon Pro 5300M / Windows + NVIDIA RTX 5070Ti*
