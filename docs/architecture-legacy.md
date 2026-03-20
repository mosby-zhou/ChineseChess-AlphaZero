# 老框架架构文档 (cchess_alphazero)

本文档描述原始 ChineseChess-AlphaZero 项目的技术架构和设计模式。

## 项目概述

`cchess_alphazero` 是一个基于 AlphaZero 算法的中国象棋强化学习项目，采用分布式架构设计，支持大规模自对弈训练。

---

## 目录结构

```
cchess_alphazero/
├── __init__.py
├── config.py              # 配置管理
├── run.py                 # 主入口
├── manager.py             # 资源管理器
├── test.py                # 测试入口
├── uci.py                 # UCI协议支持
│
├── configs/               # 配置文件
│   ├── mini.py            # 最小配置（快速测试）
│   ├── normal.py          # 标准配置
│   └── distribute.py      # 分布式配置
│
├── agent/                 # 智能体模块
│   ├── model.py           # 策略价值网络（PyTorch）
│   ├── player.py          # MCTS玩家
│   └── api.py             # 模型推理API
│
├── environment/           # 环境模块
│   ├── env.py             # 游戏环境
│   ├── chessboard.py      # 棋盘实现
│   ├── chessman.py        # 棋子定义
│   ├── static_env.py      # 静态环境（高效）
│   ├── light_env/         # 轻量级环境
│   └── lookup_tables.py   # 查找表
│
├── worker/                # 工作进程
│   ├── self_play.py       # 自对弈工作器
│   ├── optimize.py        # 训练工作器
│   ├── evaluator.py       # 评估工作器
│   ├── sl.py              # 监督学习工作器
│   └── compute_elo.py     # ELO计算
│
├── lib/                   # 工具库
│   ├── data_helper.py     # 数据处理
│   ├── model_helper.py    # 模型工具
│   ├── web_helper.py      # 网络工具
│   ├── logger.py          # 日志工具
│   └── tf_util.py         # TensorFlow工具
│
└── play_games/            # 对弈模块
    ├── play.py            # GUI对弈
    └── play_cli.py        # CLI对弈
```

---

## 核心架构

### 1. 配置系统

采用多层级配置设计，支持不同训练规模：

```python
class Config:
    def __init__(self, config_type="mini"):
        self.opts = Options()           # 全局选项
        self.resource = ResourceConfig() # 资源路径
        self.internet = InternetConfig() # 网络配置
        self.model = ModelConfig()       # 模型参数
        self.play = PlayConfig()         # 自对弈参数
        self.play_data = PlayDataConfig() # 数据配置
        self.trainer = TrainerConfig()   # 训练参数
        self.eval = EvaluateConfig()     # 评估参数
```

**配置类型**：
- `mini`: 最小配置，用于快速测试
- `normal`: 标准配置，适合单机训练
- `distribute`: 分布式配置，适合集群训练

### 2. 分布式工作器架构

```
┌─────────────────────────────────────────────────────────────┐
│                        分布式训练架构                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │ Self-Play │    │ Self-Play │    │ Self-Play │  ...       │
│  │ Worker 1  │    │ Worker 2  │    │ Worker N  │             │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘             │
│       │               │               │                    │
│       └───────────────┼───────────────┘                    │
│                       ▼                                    │
│              ┌────────────────┐                            │
│              │  Play Data Dir  │                            │
│              │  (JSON files)   │                            │
│              └───────┬────────┘                            │
│                      │                                     │
│                      ▼                                     │
│              ┌────────────────┐                            │
│              │ Optimize Worker │ ◄─── 训练模型              │
│              │   (Trainer)     │                            │
│              └───────┬────────┘                            │
│                      │                                     │
│                      ▼                                     │
│              ┌────────────────┐                            │
│              │  Best Model     │                            │
│              │  (.pth file)    │                            │
│              └───────┬────────┘                            │
│                      │                                     │
│         ┌────────────┴────────────┐                       │
│         ▼                         ▼                        │
│  ┌─────────────┐          ┌─────────────┐                 │
│  │ Evaluator   │          │ Remote Server │                │
│  │ (模型对比)   │          │ (分布式模式)   │                │
│  └─────────────┘          └─────────────┘                 │
│                                                            │
└─────────────────────────────────────────────────────────────┘
```

### 3. 核心组件

#### 3.1 策略价值网络 (agent/model.py)

```python
class PolicyValueNet(nn.Module):
    """AlphaZero风格的双头网络"""

    结构:
    - 输入层: 14×10×9 特征平面
    - 初始卷积: 将输入映射到高维空间
    - 残差塔: N个残差块（默认7个）
    - 策略头: 输出走法概率分布
    - 价值头: 输出局面评估值[-1, 1]
```

**网络参数**（标准配置）：
| 参数 | 值 |
|------|-----|
| 输入通道 | 14 |
| 卷积通道 | 256 |
| 残差块数 | 7 |
| 价值全连接 | 256 |
| 参数量 | ~100M |

#### 3.2 MCTS玩家 (agent/player.py)

```python
class CChessPlayer:
    """多线程MCTS搜索实现"""

    特点:
    - 使用线程池并行搜索
    - 虚拟损失(Virtual Loss)机制
    - 批量神经网络推理
    - 支持树复用
```

**核心数据结构**：
```python
class VisitState:
    a: dict          # 子节点 {action: ActionState}
    sum_n: int       # 总访问次数
    p: np.ndarray    # 先验概率
    legal_moves: list # 合法走法
    waiting: bool    # 是否等待网络评估

class ActionState:
    n: int   # 访问次数
    w: float # 累计价值
    q: float # 平均价值 = w/n
    p: float # 先验概率
```

#### 3.3 游戏环境 (environment/)

**两种环境实现**：

1. **标准环境** (`env.py`, `chessboard.py`)
   - 完整功能，支持GUI渲染
   - 支持棋谱记录和回放
   - 用于对弈和可视化

2. **轻量环境** (`static_env.py`, `light_env/`)
   - 高效的纯函数式实现
   - 使用FEN字符串表示状态
   - 用于自对弈训练

```python
# 状态表示: FEN字符串
# 例: "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"

# 特征平面: 14×10×9
planes = np.zeros(shape=(14, 10, 9), dtype=np.float32)
# 通道 0-6: 红方棋子 (P/C/R/N/B/A/K)
# 通道 7-13: 黑方棋子 (p/c/r/n/b/a/k)
```

---

## 设计模式

### 1. 工作器模式 (Worker Pattern)

每个工作器独立运行，通过文件系统通信：

```python
# 自对弈工作器
python cchess_alphazero/run.py self

# 训练工作器
python cchess_alphazero/run.py opt

# 评估工作器
python cchess_alphazero/run.py eval
```

### 2. 管道通信模式 (Pipe Communication)

```python
# agent/api.py - 模型推理API
class CChessModelAPI:
    """多进程推理管道"""

    def __init__(self, config, model):
        self.pipe = model.get_pipes()

    # 工作进程发送状态 -> 主进程推理 -> 返回结果
```

### 3. 虚拟损失模式 (Virtual Loss)

```python
# MCTS中的虚拟损失实现
def MCTS_search(self, state, ...):
    # 应用虚拟损失，避免重复探索
    action_state.n += virtual_loss
    action_state.w -= virtual_loss
    action_state.q = action_state.w / action_state.n

    # 回传时恢复
    action_state.n += 1 - virtual_loss
    action_state.w += v + virtual_loss
```

### 4. 生产者-消费者模式

```
Self-Play Worker (生产者)
       │
       ▼
Play Data Files (缓冲区)
       │
       ▼
Optimize Worker (消费者)
```

---

## 训练流程

### 1. 自对弈生成

```python
# worker/self_play.py
def self_play(config):
    while True:
        # 1. 加载最新模型
        model.load_best_weight()

        # 2. 执行自对弈
        game = play_game(model)

        # 3. 保存训练数据
        save_game_data(game)
```

### 2. 模型训练

```python
# worker/optimize.py
def training(self):
    while True:
        # 1. 加载新数据
        self.fill_queue()

        # 2. 训练epoch
        self.train_epoch(epochs)

        # 3. 保存模型
        self.save_current_model()

        # 4. 学习率调度
        self.update_learning_rate(total_steps)
```

### 3. 模型评估

```python
# worker/evaluator.py
def evaluate(config):
    # 对比 NextModel vs BestModel
    # 胜率 > 55% 则接受更新
```

---

## 关键技术点

### 1. 动作编码

使用预定义的动作标签列表：

```python
# environment/lookup_tables.py
ActionLabelsRed = [...]  # 红方视角的所有可能走法
# 格式: "x0y0x1y1"，如 "4041" 表示 (4,0)->(4,1)
```

### 2. 状态翻转

黑方视角通过翻转实现：

```python
def flip_move(move):
    """180度旋转走法"""
    x0, y0, x1, y1 = move
    return f"{8-x0}{9-y0}{8-x1}{9-y1}"

def flip_policy(policy):
    """翻转策略向量"""
    # 红方视角 <-> 黑方视角
```

### 3. UCI协议支持

```python
# uci.py
# 支持标准UCI协议，可与其他引擎对弈
# 命令: uci, isready, position, go, quit
```

---

## 数据流

```
┌─────────────────────────────────────────────────────────┐
│                      数据流向                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐            │
│  │  FEN    │───▶│ Planes  │───▶│ Network │            │
│  │ String  │    │ (14×10×9)│    │         │            │
│  └─────────┘    └─────────┘    └────┬────┘            │
│                                     │                  │
│                          ┌──────────┴──────────┐      │
│                          ▼                     ▼       │
│                    ┌──────────┐         ┌──────────┐  │
│                    │  Policy  │         │  Value   │  │
│                    │(走法概率) │         │(局面评估)│  │
│                    └────┬─────┘         └──────────┘  │
│                         │                              │
│                         ▼                              │
│                   ┌──────────┐                         │
│                   │   MCTS   │                         │
│                   │  搜索    │                          │
│                   └────┬─────┘                         │
│                        │                               │
│                        ▼                               │
│                  ┌──────────┐                          │
│                  │  Action  │                          │
│                  │ (最佳走法)│                          │
│                  └──────────┘                          │
│                                                        │
└─────────────────────────────────────────────────────────┘
```

---

## 依赖关系

```
requirements.txt (老框架):
- Python 3.6+
- tensorflow-gpu: 1.3.0 (已迁移到 PyTorch)
- Keras: 2.0.8 (已迁移到 PyTorch)
- numpy
- pygame (GUI)
```

---

## 使用方式

### 启动训练

```bash
# 单机训练
python cchess_alphazero/run.py self  # 终端1: 自对弈
python cchess_alphazero/run.py opt   # 终端2: 训练

# 分布式训练
python cchess_alphazero/run.py --type distribute --distributed self
python cchess_alphazero/run.py --type distribute --distributed opt
```

### 人机对弈

```bash
# GUI模式
python cchess_alphazero/run.py play

# CLI模式
python cchess_alphazero/run.py play --cli
```

---

## 局限性

1. **复杂度高**：分布式架构需要多进程协调
2. **资源需求大**：需要多GPU和大内存
3. **调试困难**：多进程环境难以调试
4. **配置繁琐**：需要手动配置多个工作器
5. **门槛高**：不适合快速实验和学习
