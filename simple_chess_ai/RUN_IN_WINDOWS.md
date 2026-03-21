# Windows 运行指南

本文档指导你在 Windows 系统上运行中国象棋 AlphaZero 项目。

---

## 目录

1. [环境准备](#1-环境准备)
2. [安装依赖](#2-安装依赖)
3. [验证安装](#3-验证安装)
4. [运行训练](#4-运行训练)
5. [运行游戏](#5-运行游戏)
6. [常见问题](#6-常见问题)

---

## 1. 环境准备

### 1.1 安装 Python

1. 访问 [Python 官网](https://www.python.org/downloads/) 下载 Python 3.10 或 3.11
   - **推荐**：Python 3.11.x（稳定且兼容性好）
   - **注意**：避免使用 Python 3.12+，部分依赖可能不兼容

2. 安装时**务必勾选**：
   - ✅ `Add Python to PATH`
   - ✅ `Install pip`

3. 验证安装：
   ```powershell
   python --version
   pip --version
   ```

### 1.2 创建虚拟环境（推荐）

```powershell
# 进入项目目录
cd ChineseChess-AlphaZero

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
.\venv\Scripts\activate

# 确认使用虚拟环境的 Python
where python
```

---

## 2. 安装依赖

### 2.1 确定你的 GPU 类型

| GPU 类型 | 推荐方案 |
|----------|----------|
| NVIDIA (GeForce/RTX) | PyTorch + CUDA |
| AMD (Radeon) | PyTorch + DirectML |
| 无独立显卡 | PyTorch + CPU |

### 2.2 NVIDIA GPU（CUDA）

```powershell
# 安装 PyTorch (CUDA 11.8 版本)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install numpy pygame tqdm pandas h5py requests
```

验证 CUDA：
```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 2.3 AMD GPU（DirectML）

```powershell
# 安装 PyTorch CPU 版本（基础依赖）
pip install torch torchvision

# 安装 DirectML 扩展
pip install torch-directml

# 安装其他依赖
pip install numpy pygame tqdm pandas h5py requests
```

验证 DirectML：
```powershell
python -c "import torch_directml as dml; print(f'DirectML Device: {dml.device_name()}')"
```

### 2.4 仅 CPU

```powershell
# 安装 PyTorch CPU 版本
pip install torch torchvision

# 安装其他依赖
pip install numpy pygame tqdm pandas h5py requests
```

---

## 3. 验证安装

### 3.1 检查 GPU 支持

```powershell
python -c "from simple_chess_ai.model import get_device; print(f'使用设备: {get_device()}')"
```

预期输出：
- NVIDIA GPU：`使用设备: cuda`
- AMD GPU：`使用设备: privateuseone:0`
- CPU：`使用设备: cpu`

### 3.2 快速测试

```powershell
# 运行快速训练测试（约 2 分钟）
python -m simple_chess_ai train --quick
```

预期输出：
```
[第 1/5 局] 胜方: 和棋, 步数: 200, ...
训练完成，平均损失: X.XX
```

---

## 4. 运行训练

### 4.1 训练命令

```powershell
# 快速训练（测试用，约 5 分钟）
python -m simple_chess_ai train --quick

# 标准训练（约 30 分钟）
python -m simple_chess_ai train --num_games 50 --num_simulations 100

# 高质量训练（数小时）
python -m simple_chess_ai train --num_games 200 --num_simulations 200

# 使用批量 MCTS（更快）
python -m simple_chess_ai train --num_games 100 --mcts_mode batch
```

### 4.2 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_games` | 50 | 自对弈局数 |
| `--num_simulations` | 100 | 每步 MCTS 模拟次数 |
| `--batch_size` | 256 | 训练批次大小 |
| `--mcts_mode` | standard | MCTS 模式：standard/optimized/batch |
| `--backend` | cnn | 网络架构：cnn（速度快）/ gnn（理论上限高）|
| `--use_fp16` | False | 启用 FP16 加速（仅 NVIDIA GPU） |
| `--use_grpo` | False | 启用 GRPO 训练方法 |

### 4.3 训练输出

训练数据保存在 `simple_chess_ai/runs/` 目录：
```
runs/
└── run_YYYYMMDD_HHMMSS/
    ├── model_best.pth      # 最佳模型
    ├── model_final.pth     # 最终模型
    ├── self_play.jsonl     # 自对弈记录
    └── training.jsonl      # 训练日志
```

---

## 5. 运行游戏

### 5.1 图形界面对弈

```powershell
# 使用默认模型
python -m simple_chess_ai play

# 使用训练好的模型
python -m simple_chess_ai play --model_path simple_chess_ai/runs/run_XXXXXX/model_best.pth

# 调整 AI 难度（模拟次数越多越强）
python -m simple_chess_ai play --num_simulations 400
```

### 5.2 图形界面操作

- **鼠标点击**：选择棋子和目标位置
- **R 键**：重新开始
- **ESC 键**：退出游戏

### 5.3 命令行对弈

```powershell
python -m simple_chess_ai play_cli --model_path path/to/model.pth
```

---

## 6. 常见问题

### Q1: `ModuleNotFoundError: No module named 'torch_directml'`

**原因**：未安装 DirectML 扩展

**解决**：
```powershell
pip install torch-directml
```

### Q2: `torch_directml` 安装失败

**原因**：Python 版本不兼容

**解决**：使用 Python 3.10 或 3.11
```powershell
# 检查 Python 版本
python --version

# 如果是 3.12+，建议重新安装 Python 3.11
```

### Q3: 训练很慢，每局超过 5 分钟

**原因**：未正确启用 GPU

**解决**：
1. 确认 GPU 被识别：
   ```powershell
   python -c "from simple_chess_ai.model import get_device; print(get_device())"
   ```
2. 如果显示 `cpu`，检查 GPU 驱动是否安装
3. AMD 用户确保已安装 [最新驱动](https://www.amd.com/en/support)

### Q4: `UserWarning: The operator '...' is not supported on the DML backend`

**原因**：DirectML 部分算子回退到 CPU

**解决**：这是**正常警告**，性能影响 <5%，可忽略。如需屏蔽：
```powershell
$env:PYTHONWARNINGS="ignore::UserWarning"
```

### Q5: PyGame 窗口打开后黑屏/闪退

**原因**：显卡驱动问题或显示器 DPI 缩放

**解决**：
1. 更新显卡驱动
2. 右键 `python.exe` → 属性 → 兼容性 → 更改高 DPI 设置 → 替代高 DPI 缩放行为

### Q6: 训练时内存不足

**原因**：批次大小过大

**解决**：
```powershell
# 减小批次大小
python -m simple_chess_ai train --batch_size 128

# 或减少模拟次数
python -m simple_chess_ai train --num_simulations 50
```

### Q7: CUDA out of memory（NVIDIA GPU）

**解决**：
```powershell
# 减小批次大小
python -m simple_chess_ai train --batch_size 128

# 或禁用 FP16
python -m simple_chess_ai train  # 不要加 --use_fp16
```

---

## 附录：完整命令速查

```powershell
# 1. 激活虚拟环境
.\venv\Scripts\activate

# 2. 验证环境
python -c "from simple_chess_ai.model import get_device; print(get_device())"

# 3. 快速测试
python -m simple_chess_ai train --quick

# 4. 正式训练（NVIDIA GPU）
python -m simple_chess_ai train --num_games 100 --num_simulations 100 --mcts_mode batch --use_fp16 --backend cnn

# 4b. 正式训练（AMD GPU - 仅支持 CNN）
python -m simple_chess_ai train --num_games 100 --num_simulations 100 --mcts_mode batch --backend cnn

# 5. 对弈
python -m simple_chess_ai play
```

---

## 技术支持

如有问题，请查看：
- 项目 README：`README.md`
- 中文说明：`README_CN.md`
- MCTS 优化文档：`simple_chess_ai/OPTIMIZATION.md`
