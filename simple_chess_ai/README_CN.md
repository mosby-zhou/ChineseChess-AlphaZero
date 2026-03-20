# 新旧项目能力对比评估报告

## 概述

本报告对比分析 `cchess_alphazero`（旧项目）与 `simple_chess_ai`（新项目）的技术架构、模型能力、训练流程，评估新项目是否能对齐和超越旧项目，以及是否符合当前业界最新趋势。

---

## 一、架构对比

### 1.1 技术栈

| 维度 | 旧项目 (cchess_alphazero) | 新项目 (simple_chess_ai) |
|------|---------------------------|--------------------------|
| 深度学习框架 | TensorFlow 1.3 + Keras 2.0.8 | PyTorch（最新版） |
| 语言版本 | Python 3.6.3 | Python 3.8+ |
| GPU 支持 | CUDA（旧版） | CUDA + DirectML（AMD） |

**评估**：新项目采用 PyTorch，具备以下优势：
- 更灵活的动态计算图，便于实验和调试
- 更好的社区支持和生态
- DirectML 支持 AMD GPU
- **结论：新项目技术栈更现代，对齐业界主流**

### 1.2 网络架构

| 维度 | 旧项目 | 新项目 |
|------|--------|--------|
| 网络类型 | 纯 CNN（ResNet） | CNN（默认）/ GNN（可选）|
| 通道数 | 256 | 128 |
| 残差块数量 | 7 | 4 |
| 参数量 | ~5M | ~2M |

**注意**：GNN 后端在 DirectML 上兼容性不佳，推荐使用 CNN 后端。

---

## 二、业界趋势符合度

| 趋势 | 旧项目 | 新项目 |
|------|--------|--------|
| **图神经网络（GNN）** | ❌ | ✅（可选）|
| **混合精度训练（FP16）** | ❌ | ✅（仅 CUDA）|
| **GRPO / 强化学习新方法** | ❌ | ✅ |
| **分布式训练** | ✅ | ❌ |
| **模型轻量化** | ❌ | ✅ |

---

## 三、总结

**新项目能否对齐旧项目？**
- 核心能力：✅ 可以对齐（规则、网络、训练流程）
- 训练规模：❌ 暂时无法对齐（缺少分布式）

**是否符合业界最新趋势？**
- ✅ 完全符合，且部分领先（GNN + GRPO）

---

*报告生成时间：2026-03-18*

---

# 用户使用手册

## 一、环境准备

### 1.1 系统要求

| 配置项 | 最低要求 | 推荐配置 |
|--------|----------|----------|
| Python | 3.8+ | 3.10+ |
| 内存 | 4GB | 8GB+ |
| 存储 | 500MB | 2GB |
| GPU | 无（可用 CPU）| AMD/NVIDIA |

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
pip install torch numpy pygame matplotlib --index-url https://download.pytorch.org/whl/cu121
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

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--num_games` | 50 | 自对弈局数 |
| `--num_simulations` | 100 | 每步 MCTS 模拟次数 |
| `--num_epochs` | 5 | 每次训练轮数 |
| `--batch_size` | 256 | 训练批大小 |
| `--lr` | 0.001 | 学习率 |
| `--max_moves` | 200 | 每局最大步数 |
| `--buffer_size` | 10000 | 数据缓冲区大小 |
| `--model_path` | saved_model/model.pth | 模型保存路径 |
| `--save_interval` | 10 | 每隔多少局保存模型 |
| `--use_grpo` | False | 使用 GRPO 训练 |
| `--grpo_group_size` | 8 | GRPO 组大小 |
| `--gating_interval` | 20 | Gating 评测间隔（0=禁用）|
| `--gating_games` | 20 | Gating 对局数 |
| `--gating_winrate` | 0.55 | Gating 接受阈值 |
| `--seed` | None | 随机种子 |
| `--quick` | False | 快速验证模式 |

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

| 项目 | 配置 |
|------|------|
| 系统 | WSL2 + Ubuntu 24.04 |
| GPU | AMD Radeon Pro 5300M |
| PyTorch | torch-directml 0.2.5 |
| 后端 | CNN（纯卷积）|

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

## 七、推荐训练配置

### AMD GPU / CPU

```bash
python -m simple_chess_ai train \
    --num_games 100 \
    --num_simulations 100 \
    --batch_size 256 \
    --gating_interval 20
```

### NVIDIA GPU（可用 FP16 加速）

```bash
python -m simple_chess_ai train \
    --num_games 100 \
    --num_simulations 200 \
    --batch_size 512 \
    --use_fp16 \
    --gating_interval 20
```

---

*使用手册版本：v2.0*
*更新时间：2026-03-18*
*测试环境：WSL2 + AMD Radeon Pro 5300M*
