# 新框架架构文档 (simple_chess_ai)

本文档描述简化版中国象棋 AI 项目的技术架构和设计模式。

## 项目概述

`simple_chess_ai` 是一个轻量级、自包含的中国象棋 AlphaZero 实现，专为研究、教学和快速原型开发而设计。采用一体化架构，无需分布式部署即可运行完整训练流程。

---

## 目录结构

```
simple_chess_ai/
├── __init__.py           # 包初始化
├── __main__.py           # 主入口 (python -m simple_chess_ai)
├── README.md             # 使用说明
│
├── game.py               # 游戏逻辑核心
│   ├── ChessGame         # 游戏状态管理
│   ├── 规则引擎          # 合法走法生成
│   └── Zobrist哈希       # 局面快速比对
│
├── model.py              # 策略价值网络
│   ├── ResBlock          # 残差块
│   ├── PolicyValueNet    # CNN网络
│   └── ChessModel        # 模型管理类
│
├── mcts.py               # 蒙特卡洛树搜索
│   ├── MCTSNode          # 树节点
│   └── MCTS              # 搜索器
│
├── train.py              # 训练管线
│   ├── self_play_game()  # 自对弈
│   ├── train_model()     # 模型训练
│   └── evaluate_models() # 模型评测
│
├── grpo.py               # GRPO训练器
│   └── GRPOTrainer       # 组相对策略优化
│
├── gnn_feature.py        # 图神经网络
│   ├── ChessGraphBuilder # 图构建器
│   └── GNNPolicyValueNet # GNN网络
│
├── reasoning.py          # 推理增强
│   └── ReasoningModule   # 思维链推理
│
├── action_encoding.py    # 动作编码
│   └── 动作空间定义
│
├── export.py             # 数据导出
│   ├── JSONL导出
│   ├── CSV导出
│   └── PNG曲线图
│
├── gui.py                # 图形界面
│   └── Pygame实现
│
├── cli.py                # 命令行界面
│
├── reasoning_cli.py      # 推理模式CLI
│
└── tests.py              # 单元测试 (34个)
```

---

## 核心架构

### 1. 一体化架构

```
┌─────────────────────────────────────────────────────────────┐
│                    一体化训练架构                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    train.py                          │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐         │   │
│  │  │ 自对弈   │───▶│ 数据缓冲 │───▶│ 网络训练 │         │   │
│  │  │(MCTS)   │    │  (队列)  │    │(PyTorch)│         │   │
│  │  └─────────┘    └─────────┘    └─────────┘         │   │
│  │       │                               │              │   │
│  │       │         ┌─────────┐           │              │   │
│  │       └────────▶│ 模型评测 │◀──────────┘              │   │
│  │                 │(Gating) │                          │   │
│  │                 └────┬────┘                          │   │
│  └──────────────────────┼───────────────────────────────┘   │
│                         │                                   │
│                         ▼                                   │
│                  ┌──────────┐                               │
│                  │ 模型保存 │                               │
│                  │  (.pth)  │                               │
│                  └──────────┘                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. 核心组件

#### 2.1 游戏引擎 (game.py)

```python
class ChessGame:
    """自包含的中国象棋游戏类"""

    核心功能:
    - 完整规则实现（蹩马腿、塞象眼、飞将）
    - 合法走法生成（含走后将军过滤）
    - 胜负判断（含长将、长捉检测）
    - Zobrist哈希（快速局面比对）
    - FEN表示（标准棋谱格式）
```

**关键特性**：

```python
# 走后将军过滤
def get_legal_moves(self):
    moves = []
    for move in self._get_piece_moves():
        if not self._move_leaves_king_in_check(move):
            moves.append(move)  # 只保留不送将的走法
    return moves

# Zobrist哈希
class ChessGame:
    def __init__(self):
        self.pos_hash = 0  # 局面哈希值
        self.pos_history = []  # 历史哈希（用于检测重复）

    def _zobrist_piece(self, piece, x, y):
        """计算棋子对哈希的贡献"""
        return _ZOBRIST_PIECES[PIECE_TO_ZOBRIST_IDX[piece]][y * 9 + x]
```

#### 2.2 神经网络 (model.py)

```python
class PolicyValueNet(nn.Module):
    """简化版策略价值网络"""

    结构:
    - 输入: 14×10×9 特征平面
    - 初始卷积: 14 → 128 通道
    - 残差塔: 4个残差块
    - 策略头: 输出 NUM_ACTIONS 维 logits
    - 价值头: 输出 1 维评估值
```

**关键改进**：

```python
class ChessModel:
    def predict_with_mask(self, planes, legal_indices):
        """
        带掩码的预测，避免非法走法污染概率分布

        相比原项目的改进：
        1. 先将非法走法 logit 置为极小值
        2. 再做 softmax，保证概率质量仅在合法走法上
        """
        logits, value = self.model(tensor)
        mask = torch.full_like(logits, -1e9)
        mask[0, legal_indices] = 0.0
        logits = logits + mask
        policy = F.softmax(logits, dim=1)
        return policy, value
```

#### 2.3 MCTS搜索 (mcts.py)

```python
class MCTS:
    """简化的蒙特卡洛树搜索"""

    特点:
    - 支持Dirichlet噪声注入（训练时探索）
    - 支持树复用（减少重复计算）
    - 支持局面缓存（加速搜索）
    - 可配置温度参数
```

**两种运行模式**：

```python
# 模式1: 每次重置树（默认，适合推理）
mcts.get_action_probs(game, reset_root=True)

# 模式2: 树复用（适合训练循环）
mcts.get_action_probs(game, reset_root=False)
mcts.update_with_move(action)  # 手动推进root
```

**Dirichlet噪声注入**（2026年修复）：

```python
def _add_dirichlet_noise(self, node):
    """
    正确注入位置：第一次扩展root后
    公式: prior = (1-w) * prior + w * noise
    """
    noise = np.random.dirichlet([alpha] * n_children)
    for child, eta in zip(node.children.values(), noise):
        child.prior = (1 - weight) * child.prior + weight * eta
```

#### 2.4 GRPO训练 (grpo.py)

```python
class GRPOTrainer:
    """
    Group Relative Policy Optimization

    参考 DeepSeek R1 的训练思路：
    - 对同一局面采样多个动作
    - 计算组内相对优势作为奖励基准
    - 无需 Critic 网络，减少内存开销
    """

    def compute_advantages(self, rewards):
        # 组内相对优势
        mean = rewards.mean()
        std = rewards.std() + 1e-8
        return (rewards - mean) / std
```

#### 2.5 GNN特征提取 (gnn_feature.py)

```python
class ChessGraphBuilder:
    """将棋盘转换为图结构"""

    设计:
    - 90个节点（棋盘位置）
    - 边表示攻击/防守关系
    - 使用GAT学习节点间注意力权重

class GNNPolicyValueNet(nn.Module):
    """GNN增强的策略价值网络"""

    结构:
    - CNN分支: 提取局部特征
    - GNN分支: 建模全局关系
    - 特征融合: 拼接后输出策略/价值
```

---

## 设计模式

### 1. 策略模式 (Strategy Pattern)

```python
# 训练后端可选
class ChessModel:
    def build(self):
        if self.backend == 'gnn':
            self.model = GNNPolicyValueNet(...)
        else:
            self.model = PolicyValueNet(...)
```

### 2. 模板方法模式 (Template Method)

```python
# 训练流程模板
def train_loop(self):
    while not done:
        data = self.self_play()      # 步骤1
        self.update_buffer(data)     # 步骤2
        self.train_step()            # 步骤3
        if self.should_evaluate():   # 步骤4
            self.gating_evaluate()
```

### 3. 建造者模式 (Builder Pattern)

```python
# CLI参数构建
parser = argparse.ArgumentParser()
parser.add_argument('--num_games', type=int, default=50)
parser.add_argument('--use_grpo', action='store_true')
parser.add_argument('--use_fp16', action='store_true')
# ... 更多参数
```

---

## 训练流程

### 标准训练流程

```python
def train():
    for game_idx in range(num_games):
        # 1. 自对弈
        data, winner, moves = self_play_game(model, num_simulations)

        # 2. 数据收集
        replay_buffer.extend(data)

        # 3. 网络训练
        if len(replay_buffer) > batch_size:
            train_model(model, replay_buffer, epochs)

        # 4. Gating评测（可选）
        if gating_interval > 0 and game_idx % gating_interval == 0:
            score = evaluate_models(candidate, best, gating_games)
            if score > gating_winrate:
                best = candidate
            else:
                candidate = best  # 回滚
```

### GRPO训练流程

```python
def train_grpo():
    for game_idx in range(num_games):
        # 1. 组采样：同一局面采样多个动作
        group_data = sample_action_group(model, game, group_size)

        # 2. 计算相对优势
        advantages = compute_group_advantages(group_data)

        # 3. 策略优化
        grpo_optimizer.step(advantages)
```

---

## 数据流

```
┌─────────────────────────────────────────────────────────────┐
│                      数据流向                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐                                            │
│  │ ChessGame   │                                            │
│  │ (当前局面)  │                                            │
│  └──────┬──────┘                                            │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────┐     ┌─────────────┐                       │
│  │ to_planes() │────▶│ 14×10×9     │                       │
│  │             │     │ 特征平面    │                        │
│  └─────────────┘     └──────┬──────┘                       │
│                             │                               │
│         ┌───────────────────┼───────────────────┐          │
│         ▼                   ▼                   ▼          │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐  │
│  │ CNN分支     │     │ GNN分支     │     │ 推理模块    │  │
│  │(局部特征)   │     │(全局关系)   │     │(CoT分析)    │  │
│  └──────┬──────┘     └──────┬──────┘     └─────────────┘  │
│         │                   │                               │
│         └─────────┬─────────┘                               │
│                   ▼                                         │
│           ┌─────────────┐                                   │
│           │ 特征融合    │                                   │
│           └──────┬──────┘                                   │
│                  │                                          │
│         ┌────────┴────────┐                                 │
│         ▼                 ▼                                 │
│  ┌─────────────┐   ┌─────────────┐                         │
│  │ 策略头      │   │ 价值头      │                         │
│  │(走法概率)   │   │(局面评估)   │                         │
│  └──────┬──────┘   └──────┬──────┘                         │
│         │                 │                                 │
│         └────────┬────────┘                                 │
│                  ▼                                          │
│           ┌─────────────┐                                   │
│           │    MCTS     │                                   │
│           │   搜索      │                                   │
│           └──────┬──────┘                                   │
│                  │                                          │
│                  ▼                                          │
│           ┌─────────────┐                                   │
│           │ 最佳走法    │                                   │
│           └─────────────┘                                   │
│                                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 新增特性

### 1. 模型评测门控 (Gating)

```python
def evaluate_models(model_a, model_b, n_games=20):
    """
    通过对局评测两个模型

    特点:
    - 禁用噪声，确定性搜索
    - 交替红黑方，减少先手偏差
    - draw=0.5计分
    """
    score = (wins_a + 0.5 * draws) / total
    return score
```

### 2. FP16混合精度

```python
# train.py
def train_model(..., use_fp16=False):
    if use_fp16:
        scaler = torch.amp.GradScaler()
        with torch.amp.autocast('cuda'):
            loss = compute_loss()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### 3. 数据导出管道

```python
# export.py
def init_run_dir(base_dir):
    """创建带时间戳的运行目录"""
    run_dir = base_dir / f"run_{timestamp}"
    # 创建配置文件、数据文件、图表目录

def append_self_play_jsonl(run_dir, data):
    """追加自对弈记录（JSONL格式）"""

def plot_curves(run_dir):
    """生成训练曲线图（PNG）"""
```

### 4. 推理增强 (CoT)

```python
# reasoning.py
class ReasoningModule:
    """思维链推理"""

    def analyze(self, game):
        return {
            'threats': self.find_threats(game),
            'material': self.evaluate_material(game),
            'recommendation': self.recommend_move(game),
            'reasoning': self.generate_cot(game),
        }
```

---

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_games` | 50 | 自对弈局数 |
| `--num_simulations` | 100 | MCTS模拟次数 |
| `--num_epochs` | 5 | 每局训练轮数 |
| `--batch_size` | 256 | 批大小 |
| `--lr` | 0.001 | 学习率 |
| `--buffer_size` | 10000 | 数据缓冲大小 |
| `--use_grpo` | False | 使用GRPO训练 |
| `--use_fp16` | False | 使用FP16混合精度 |
| `--gating_interval` | 20 | Gating评测间隔 |
| `--gating_winrate` | 0.55 | Gating接受阈值 |
| `--seed` | None | 随机种子 |

---

## 与老框架对比

| 特性 | 老框架 (cchess_alphazero) | 新框架 (simple_chess_ai) |
|------|---------------------------|--------------------------|
| 架构 | 分布式多进程 | 单进程一体化 |
| 网络规模 | 256通道×7块 (~100M) | 128通道×4块 (~25M) |
| 训练模式 | 标准AlphaZero | AlphaZero + GRPO |
| 后端 | CNN | CNN + GNN可选 |
| 精度 | FP32 | FP32/FP16可选 |
| 数据导出 | 无 | JSONL/CSV/PNG |
| 测试覆盖 | 无 | 34个单元测试 |
| 使用门槛 | 高（需配置多进程） | 低（单命令运行） |
| 适用场景 | 大规模生产训练 | 研究原型开发 |

---

## 使用方式

### 训练

```bash
# 快速训练
python -m simple_chess_ai train --num_games 100

# GRPO + FP16
python -m simple_chess_ai train --use_grpo --use_fp16

# 带Gating
python -m simple_chess_ai train --gating_interval 20 --gating_winrate 0.55
```

### 对弈

```bash
# GUI模式
python -m simple_chess_ai play

# CLI模式
python -m simple_chess_ai play_cli

# 推理模式（显示AI思考过程）
python -m simple_chess_ai reason
```

### 测试

```bash
python -m simple_chess_ai.tests
```
