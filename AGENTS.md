# AGENTS.md - AI 编码指南

本文档为 AI 编码助手提供项目上下文和编码规范，确保生成代码与项目风格一致。

---

## 项目概述

ChineseChess-AlphaZero 是一个基于 AlphaZero 算法的中国象棋强化学习项目。项目包含两套代码：

1. **cchess_alphazero/** - 老框架，分布式架构，适合大规模生产训练
2. **simple_chess_ai/** - 新框架，一体化架构，适合研究和快速原型

---

## 快速参考

### 常用命令

```bash
# 新框架训练
python -m simple_chess_ai train --num_games 100

# 新框架对弈
python -m simple_chess_ai play

# 运行测试
python -m simple_chess_ai.tests

# 老框架自对弈
python cchess_alphazero/run.py self

# 老框架训练
python cchess_alphazero/run.py opt
```

### 关键文件路径

```
simple_chess_ai/
├── game.py          # 游戏规则引擎
├── model.py         # 策略价值网络
├── mcts.py          # MCTS搜索
├── train.py         # 训练管线
├── grpo.py          # GRPO训练器
├── gnn_feature.py   # GNN特征
└── tests.py         # 单元测试

cchess_alphazero/
├── agent/model.py   # 策略价值网络
├── agent/player.py  # MCTS玩家
├── worker/optimize.py # 训练工作器
└── environment/     # 游戏环境
```

---

## 编码规范

### 1. 语言规范

- **代码注释**：使用中文注释，便于理解
- **文档字符串**：使用中文描述，格式如下：

```python
def function_name(arg1, arg2):
    """
    函数简述

    详细描述...

    Args:
        arg1: 参数1说明
        arg2: 参数2说明

    Returns:
        返回值说明
    """
```

### 2. 命名规范

```python
# 类名：大驼峰
class ChessGame:
class MCTSNode:
class PolicyValueNet:

# 函数/方法：蛇形命名
def get_legal_moves():
def compute_advantages():
def train_model():

# 常量：全大写下划线
BOARD_HEIGHT = 10
BOARD_WIDTH = 9
NUM_ACTIONS = 2086

# 私有方法：下划线前缀
def _get_piece_moves():
def _add_dirichlet_noise():
def _compute_hash():
```

### 3. 导入顺序

```python
# 1. 标准库
import os
import json
import copy
from collections import deque

# 2. 第三方库
import numpy as np
import torch
import torch.nn as nn

# 3. 项目内部模块
from simple_chess_ai.game import ChessGame, NUM_ACTIONS
from simple_chess_ai.model import ChessModel
```

### 4. 类型标注

```python
# 推荐使用类型标注
def predict(self, planes: np.ndarray) -> tuple[np.ndarray, float]:
    ...

def get_legal_moves(self, side: str | None = None) -> list[str]:
    ...
```

---

## 核心数据结构

### 1. 棋盘表示

```python
# FEN字符串
FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"

# 二维数组 (10行 × 9列)
board = [[None] * 9 for _ in range(10)]
# board[y][x] = 棋子字符 或 None
# 大写=红方, 小写=黑方
# R/r=车, N/n=马, B/b=象, A/a=仕, K/k=将帅, C/c=炮, P/p=兵卒

# 特征平面 (14 × 10 × 9)
planes = np.zeros((14, 10, 9), dtype=np.float32)
# 通道 0-6: 红方棋子 (P, C, R, N, B, A, K)
# 通道 7-13: 黑方棋子 (p, c, r, n, b, a, k)
```

### 2. 动作编码

```python
# 走法格式: "x0y0x1y1"
# x: 列 0-8, y: 行 0-9
# 例: "4041" = (4,0) -> (4,1) 帅向前一步

# 动作空间大小
NUM_ACTIONS = 2086  # 所有可能走法

# 翻转换位（黑方视角转红方视角）
def flip_move(move: str) -> str:
    x0, y0, x1, y1 = int(move[0]), int(move[1]), int(move[2]), int(move[3])
    return f"{8-x0}{9-y0}{8-x1}{9-y1}"
```

### 3. MCTS节点

```python
# 新框架
class MCTSNode:
    visit_count: int      # 访问次数 N
    total_value: float    # 累计价值 W
    prior: float          # 先验概率 P
    children: dict        # 子节点 {action: MCTSNode}

    @property
    def q_value(self):    # Q = W / N
        return self.total_value / self.visit_count

# 老框架
class VisitState:
    a: dict               # {action: ActionState}
    sum_n: int            # 总访问次数
    p: np.ndarray         # 先验概率
    legal_moves: list     # 合法走法

class ActionState:
    n: int    # N(s,a)
    w: float  # W(s,a)
    q: float  # Q(s,a) = W/N
    p: float  # P(s,a)
```

---

## 关键算法实现

### 1. PUCT公式

```python
def _select_child(self, node):
    """PUCT选择"""
    sqrt_total = math.sqrt(node.visit_count)

    for action, child in node.children.items():
        q = child.q_value
        u = self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
        score = q + u

        # 选择最高分
```

### 2. Dirichlet噪声

```python
def _add_dirichlet_noise(self, node):
    """向root节点注入噪声，增强探索"""
    noise = np.random.dirichlet([alpha] * n_children)
    for child, eta in zip(node.children.values(), noise):
        child.prior = (1 - weight) * child.prior + weight * eta
    # 默认: alpha=0.3, weight=0.25
```

### 3. 带掩码的预测

```python
def predict_with_mask(self, planes, legal_indices):
    """避免非法走法污染概率分布"""
    logits, value = self.model(tensor)

    # 将非法走法logit置为极小值
    mask = torch.full_like(logits, -1e9)
    mask[0, legal_indices] = 0.0
    logits = logits + mask

    policy = F.softmax(logits, dim=1)
    return policy, value
```

### 4. 温度采样

```python
def apply_temperature(visits, temperature):
    """根据温度参数从访问次数计算概率"""
    if temperature < 1e-8:
        # 确定性选择
        best_idx = np.argmax(visits)
        probs = np.zeros_like(visits)
        probs[best_idx] = 1.0
    else:
        # 随机采样
        visits_temp = visits ** (1.0 / temperature)
        probs = visits_temp / visits_temp.sum()
    return probs
```

---

## 常见修改场景

### 场景1: 添加新棋子类型

1. 修改 `game.py` 中的 `PIECE_TO_INDEX`
2. 更新 `create_action_labels()` 添加新走法
3. 修改 `_get_piece_moves()` 添加移动规则
4. 更新 `NUM_ACTIONS` 常量

### 场景2: 修改网络结构

1. 修改 `model.py` 中的 `PolicyValueNet`
2. 调整通道数、残差块数
3. 更新 `save()/load()` 处理新参数
4. 运行测试验证

### 场景3: 添加新训练模式

1. 在 `train.py` 添加新的训练函数
2. 参考 `grpo.py` 实现
3. 添加CLI参数支持
4. 更新 `export.py` 导出逻辑

### 场景4: 添加新规则

1. 修改 `game.py` 中的 `_get_piece_moves()`
2. 添加胜负判断逻辑到 `step()` 方法
3. 更新 `get_legal_moves()` 过滤逻辑
4. 添加单元测试

---

## 测试规范

### 单元测试位置

```python
# simple_chess_ai/tests.py
class TestGame(unittest.TestCase):
    def test_initial_position(self): ...
    def test_legal_moves(self): ...
    def test_checkmate(self): ...

class TestModel(unittest.TestCase):
    def test_forward_pass(self): ...
    def test_save_load(self): ...

class TestMCTS(unittest.TestCase):
    def test_search(self): ...
    def test_dirichlet_noise(self): ...
```

### 测试运行

```bash
# 运行所有测试
python -m simple_chess_ai.tests

# 运行单个测试类
python -m unittest simple_chess_ai.tests.TestGame

# 详细输出
python -m simple_chess_ai.tests -v
```

---

## 性能优化提示

### 1. 减少内存分配

```python
# 不推荐：每次创建新数组
def to_planes(self):
    planes = np.zeros((14, 10, 9))  # 每次分配

# 推荐：预分配缓冲区
class ChessGame:
    def __init__(self):
        self._planes_buffer = np.zeros((14, 10, 9), dtype=np.float32)

    def to_planes(self):
        self._planes_buffer.fill(0)  # 重用
        # 填充数据
        return self._planes_buffer
```

### 2. 批量推理

```python
# 不推荐：单次推理
for state in states:
    policy, value = model.predict(state)

# 推荐：批量推理
policies, values = model.predict_batch(states)
```

### 3. 树复用

```python
# 训练时复用MCTS树
mcts = MCTS(model)
for move in game:
    actions, probs = mcts.get_action_probs(game, reset_root=False)
    game.step(action)
    mcts.update_with_move(action)  # 推进树
```

---

## 调试技巧

### 1. 打印棋盘

```python
def print_board(self):
    """打印棋盘状态"""
    pieces = {
        'R': '車', 'N': '馬', 'B': '象', 'A': '仕', 'K': '帥',
        'C': '炮', 'P': '兵',
        'r': '车', 'n': '马', 'b': '象', 'a': '士', 'k': '将',
        'c': '砲', 'p': '卒',
    }
    for y in range(9, -1, -1):
        row = ''
        for x in range(9):
            p = self.board[y][x]
            row += pieces.get(p, '·')
        print(row)
```

### 2. 跟踪MCTS

```python
# 在MCTSNode中添加调试信息
class MCTSNode:
    def debug_info(self, action):
        return f"{action}: N={self.visit_count}, Q={self.q_value:.3f}, P={self.prior:.3f}"
```

### 3. 检查走法合法性

```python
def validate_move(self, move):
    """验证走法是否合法"""
    legal = self.get_legal_moves()
    if move not in legal:
        print(f"Illegal move: {move}")
        print(f"Legal moves: {legal}")
        return False
    return True
```

---

## 常见错误

### 1. 状态不一致

```python
# 错误：直接修改棋盘后忘记更新哈希
game.board[y][x] = piece  # 错误

# 正确：使用step方法
game.step(move)  # 会自动更新哈希和历史
```

### 2. 视角混淆

```python
# 错误：黑方走棋时使用红方视角
action = mcts.get_action(game)  # 返回红方视角
game.step(action)  # 黑方视角执行会出错

# 正确：翻转换位
if not game.red_to_move:
    actual_action = flip_move(action)
game.step(actual_action)
```

### 3. 内存泄漏

```python
# 错误：MCTS树不断增长
mcts = MCTS(model)
for _ in range(10000):
    mcts.get_action_probs(game)
    # 不重置root，树无限增长

# 正确：定期重置
mcts.root = MCTSNode()  # 每局重置
```

---

## 文档索引

| 文档 | 路径 | 内容 |
|------|------|------|
| 2026改动分析 | `docs/2026-changes-analysis.md` | 新版本改动详情 |
| 老框架架构 | `docs/architecture-legacy.md` | 分布式架构设计 |
| 新框架架构 | `docs/architecture-simple.md` | 一体化架构设计 |
| 新框架README | `simple_chess_ai/README.md` | 使用说明和API |
| 老框架README | `README.md` | 项目总览 |

---

## 联系方式

如有问题，请通过 GitHub Issues 反馈。
