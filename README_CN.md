# 中国象棋Zero（CCZero）

<a href="https://cczero.org">
  <img src="https://cczero.org/zero2.png" alt="应用图标" />
</a>

## 简介

基于 [AlphaZero](https://arxiv.org/abs/1712.01815) 方法的中国象棋强化学习项目。

本项目基于以下主要资源：
1. DeepMind 10月19日发表的论文：[Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)
2. @mokemokechicken/@Akababa/@TDteach 在其仓库中对 DeepMind 思想的出色 Reversi/象棋/中国象棋实现：
   - https://github.com/mokemokechicken/reversi-alpha-zero
   - https://github.com/Akababa/Chess-Zero
   - https://github.com/TDteach/AlphaZero_ChineseChess
3. 带图形界面的中国象棋引擎：https://github.com/mm12432/MyChess


## 参与训练

为了构建一个与 AlphaZero 技术相当的强大中国象棋 AI，我们需要通过分布式项目来实现，因为这需要大量的计算资源。

如果您想加入我们一起构建世界上最强的中国象棋 AI：

* 操作指南请参见 [wiki](https://github.com/NeymarL/ChineseChess-AlphaZero/wiki)
* 实时状态请参见 https://cczero.org

![elo](elo.png)


## 环境

* Python 3.6.3
* tensorflow-gpu: 1.3.0
* Keras: 2.0.8


## 模块

### 强化学习

本 AlphaZero 实现由两个工作进程组成：`self` 和 `opt`。

* `self` 是自对弈进程，使用最佳模型进行自对弈生成训练数据。
* `opt` 是训练进程，训练模型并生成新模型。

为了加快训练速度，还引入了另外两个工作进程：

* `sl` 是监督学习进程，用于训练从网络爬取的数据。
* `eval` 是评估进程，用于评估新一代模型与当前最佳模型的对比。

### 内置图形界面

依赖：pygame

```bash
python cchess_alphazero/run.py play
```

**截图**

![board](screenshots/board.png)

您可以选择不同的棋盘/棋子样式和方向，参见[人机对弈](#人机对弈)。


## 使用方法

### 安装

### 安装依赖库
```bash
pip install -r requirements.txt
```

如果只想使用 CPU，请在 `requirements.txt` 中将 `tensorflow-gpu` 替换为 `tensorflow`。

确保 Keras 使用 TensorFlow 后端，并且 Python 版本为 3.6.3+。

### 配置

**PlayDataConfig**

* `nb_game_in_file, max_file_num`：训练数据的最大对局数为 `nb_game_in_file * max_file_num`。

**PlayConfig, PlayWithHumanConfig**

* `simulation_num_per_move`：每步的 MCTS 模拟次数。
* `c_puct`：MCTS 中价值网络和策略网络的平衡参数。
* `search_threads`：MCTS 中速度和准确性的平衡参数。
* `dirichlet_alpha`：自对弈中的随机参数。

### 完整用法

```
usage: run.py [-h] [--new] [--type TYPE] [--total-step TOTAL_STEP]
              [--ai-move-first] [--cli] [--gpu GPU] [--onegreen] [--skip SKIP]
              [--ucci] [--piece-style {WOOD,POLISH,DELICATE}]
              [--bg-style {CANVAS,DROPS,GREEN,QIANHONG,SHEET,SKELETON,WHITE,WOOD}]
              [--random {none,small,medium,large}] [--distributed] [--elo]
              {self,opt,eval,play,eval,sl,ob}

positional arguments:
  {self,opt,eval,play,eval,sl,ob}
                        执行的操作类型

optional arguments:
  -h, --help            显示帮助信息并退出
  --new                 从新的最佳模型开始运行
  --type TYPE           使用指定配置类型
  --total-step TOTAL_STEP
                        设置 TrainerConfig.start_total_steps
  --ai-move-first       设置 AI 或人类先手
  --cli                 使用命令行与 AI 对弈，默认使用图形界面
  --gpu GPU             GPU 设备列表
  --onegreen            使用 onegreen 数据训练监督学习
  --skip SKIP           跳过的对局数
  --ucci                使用 UCCI 引擎而非自对弈
  --piece-style {WOOD,POLISH,DELICATE}
                        选择棋子样式
  --bg-style {CANVAS,DROPS,GREEN,QIANHONG,SHEET,SKELETON,WHITE,WOOD}
                        选择棋盘样式
  --random {none,small,medium,large}
                        选择随机性程度
  --distributed         是否从远程服务器上传/下载文件
  --elo                 是否计算 ELO 等级分
```

### 自对弈

```
python cchess_alphazero/run.py self
```

执行后，将使用最佳模型开始自对弈。如果最佳模型不存在，将创建新的随机模型并设为最佳模型。自对弈记录存储在 `data/play_record`，最佳模型存储在 `data/model`。

选项

* `--new`：创建新的最佳模型
* `--type mini`：使用迷你配置（参见 `cchess_alphazero/configs/mini.py`）
* `--gpu '1'`：指定使用的 GPU
* `--ucci`：是否使用 UCCI 引擎对战（而非自对弈，参见 `cchess_alphazero/worker/play_with_ucci_engine.py`）
* `--distributed`：以分布式模式运行自对弈，会将自对弈数据上传到远程服务器并从服务器下载最新模型

**注意1**：要参与训练，您应该运行 `python cchess_alphazero/run.py --type distribute --distributed self`（并且不要修改配置文件 `configs/distribute.py`），更多信息请参见 [wiki](https://github.com/NeymarL/ChineseChess-AlphaZero/wiki/For-Developers)。

**注意2**：如果想在图形界面中查看自对弈记录，请参见 [wiki](https://github.com/NeymarL/ChineseChess-AlphaZero/wiki/View-self-play-games-in-GUI)。

### 训练器

```
python cchess_alphazero/run.py opt
```

执行后，将开始训练。会加载当前的最佳模型。每个 epoch 训练后的模型会保存为新的最佳模型。

选项

* `--type mini`：使用迷你配置（参见 `cchess_alphazero/configs/mini.py`）
* `--total-step TOTAL_STEP`：指定总步数（mini-batch）。总步数影响训练的学习率。
* `--gpu '1'`：指定使用的 GPU

**在 Tensorboard 中查看训练日志**

```
tensorboard --logdir logs/
```

然后访问 `http://<机器IP>:6006/`。

### 人机对弈

**使用内置图形界面**

```
python cchess_alphazero/run.py play
```

执行后，会加载最佳模型与人类对弈。

选项

* `--ai-move-first`：如果设置此选项，AI 先手，否则人类先手。
* `--type mini`：使用迷你配置（参见 `cchess_alphazero/configs/mini.py`）
* `--gpu '1'`：指定使用的 GPU
* `--piece-style WOOD`：选择棋子样式，默认为 `WOOD`
* `--bg-style CANVAS`：选择棋盘样式，默认为 `CANVAS`
* `--cli`：如果设置此标志，在命令行环境而非图形界面与 AI 对弈

**注意**：开始之前，您需要下载/找一个字体文件（`.ttc`）并将其重命名为 `PingFang.ttc`，然后放入 `cchess_alphazero/play_games` 目录。我已从此仓库中移除了字体文件因为它太大，但您可以从[这里](http://alphazero.52coding.com.cn/PingFang.ttc)下载。

您也可以直接从[这里](https://pan.baidu.com/s/1uE_zmkn0x9Be_olRL9U9cQ)下载 Windows 可执行文件。更多信息请参见 [wiki](https://github.com/NeymarL/ChineseChess-AlphaZero/wiki/For-Non-Developers#%E4%B8%8B%E6%A3%8B)。

**UCI 模式**

```
python cchess_alphazero/uci.py
```

如果您想在"冰河五四"等通用图形界面中对弈，可以从[这里](https://share.weiyun.com/5cK50Z4)下载 Windows 可执行文件。更多信息请参见 [wiki](https://github.com/NeymarL/ChineseChess-AlphaZero/wiki/For-Non-Developers#%E4%B8%8B%E6%A3%8B)。

### 评估器

```
python cchess_alphazero/run.py eval
```

执行后，用新一代模型与当前最佳模型进行对比评估。如果新一代模型不存在，工作进程会等待直到它出现，每5分钟检查一次。

选项

* `--type mini`：使用迷你配置（参见 `cchess_alphazero/configs/mini.py`）
* `--gpu '1'`：指定使用的 GPU

### 监督学习

```
python cchess_alphazero/run.py sl
```

执行后，将开始训练。会加载当前的 SLBestModel。每个 epoch 训练后的模型会保存为新的 SLBestModel。

*关于数据*

我有两个数据来源，一个是从 https://wx.jcloud.com/market/packet/10479 下载的；另一个是从 http://game.onegreen.net/chess/Index.html 爬取的（使用 --onegreen 选项）。

选项

* `--type mini`：使用迷你配置（参见 `cchess_alphazero/configs/mini.py`）
* `--gpu '1'`：指定使用的 GPU
* `--onegreen`：如果设置此标志，`sl_onegreen` 工作进程将开始训练从 `game.onegreen.net` 爬取的数据
* `--skip SKIP`：如果设置此标志，索引小于 `SKIP` 的对局将不会被用于训练（仅在设置了 `onegreen` 标志时有效）

---

## 简易象棋 AI（`simple_chess_ai`）

`simple_chess_ai` 是一个轻量级、自包含的中国象棋AI模块，基于PyTorch，无需Keras/TensorFlow即可独立运行。适合在普通CPU/GPU机器上快速训练与调试。

### 特性

- **完整象棋规则**：实现所有棋子移动规则，包括长将判负、**长捉判负**（perpetual chase）、三次重复局面判和等。
- **策略价值网络**：残差卷积网络（支持纯CNN或GNN后端），输入14通道特征平面，输出走法概率与局面价值。
- **MCTS搜索**：PUCT算法驱动，支持Dirichlet噪声、树复用（Tree Reuse）、局面缓存。
- **多种训练模式**：标准AlphaZero式自对弈训练、GRPO（Group Relative Policy Optimization）训练、FP16混合精度训练。
- **Gating评测**：定期用新模型与旧模型对局，胜率超过阈值后替换best模型。
- **数据导出**：自对弈记录（JSONL）、训练指标（CSV）、损失/胜率曲线（PNG）。
- **图形界面**：基于pygame的可交互棋盘（`python -m simple_chess_ai play`）。
- **命令行界面**：纯文本人机对弈（`python -m simple_chess_ai play_cli`）。
- **Reasoning模块**：链式推理（Chain-of-Thought）增强的走法分析。

### 快速开始

#### 安装依赖

```bash
pip install torch numpy
pip install pygame  # 仅图形界面需要
```

#### 训练模型

```bash
# 标准训练（50局自对弈）
python -m simple_chess_ai train --num_games 50 --num_simulations 100

# GRPO训练模式
python -m simple_chess_ai train --num_games 50 --use_grpo --grpo_group_size 8

# FP16混合精度训练（需要GPU）
python -m simple_chess_ai train --num_games 100 --use_fp16

# 快速验证训练流程
python -m simple_chess_ai train --quick
```

完整训练选项：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_games` | 50 | 自对弈局数 |
| `--num_simulations` | 100 | 每步MCTS模拟次数 |
| `--num_epochs` | 5 | 每次训练轮数 |
| `--batch_size` | 256 | 批大小 |
| `--lr` | 0.001 | 学习率 |
| `--max_moves` | 200 | 每局最大步数（超过判和）|
| `--buffer_size` | 10000 | 训练数据缓冲区大小 |
| `--model_path` | — | 模型保存路径（默认 `simple_chess_ai/saved_model/model.pth`）|
| `--save_interval` | 10 | 每隔多少局保存模型 |
| `--use_grpo` | — | 使用GRPO训练模式 |
| `--grpo_group_size` | 8 | GRPO组采样大小 |
| `--use_fp16` | — | FP16混合精度训练 |
| `--gating_interval` | 20 | 每隔多少局进行gating评测（0=禁用）|
| `--gating_games` | 20 | gating对局数 |
| `--gating_winrate` | 0.55 | gating接受阈值（新模型最低胜率）|
| `--seed` | — | 随机种子（可复现）|
| `--deterministic` | — | cuDNN确定性模式（配合`--seed`）|
| `--runs_dir` | — | 日志导出目录（默认 `simple_chess_ai/runs/`）|
| `--quick` | — | 快速模式（1局+1次更新，验证流程）|

#### 图形界面对弈

```bash
python -m simple_chess_ai play [--model_path path/to/model.pth] [--num_simulations 200]
```

#### 命令行对弈

```bash
python -m simple_chess_ai play_cli [--model_path path/to/model.pth] [--num_simulations 200] [--human_color red|black]
```

走法格式：`x0 y0 x1 y1`，例如 `4 0 4 1` 表示帅从(4,0)向前走一步。

### 象棋规则说明

坐标系：x 为列(0–8)，y 为行(0–9)；红方在下方(y=0–4)，黑方在上方(y=5–9)。

棋子FEN编码：
- 大写=红方：`R`(车) `N`(马) `B`(象) `A`(仕) `K`(帅) `C`(炮) `P`(兵)
- 小写=黑方：`r`(车) `n`(马) `b`(象) `a`(仕) `k`(将) `c`(炮) `p`(卒)

重复局面处理：
- **长将**（perpetual check）：循环内一方连续将军维持循环 → 该方判负
- **长捉**（perpetual chase）：循环内一方始终用棋子威胁吃对方某一非将棋子，而对方无捉回 → 捉子方判负
- **三次重复且无长将/长捉**：判和

### 运行测试

```bash
python -m pytest simple_chess_ai/tests.py -v
```

### 模块结构

```
simple_chess_ai/
├── game.py           # 象棋规则、走法生成、长将/长捉判断
├── model.py          # 策略价值网络（CNN + 残差块）
├── mcts.py           # MCTS搜索（PUCT算法）
├── train.py          # 自对弈训练流程
├── grpo.py           # GRPO训练器
├── gnn_feature.py    # 图神经网络特征提取
├── reasoning.py      # 链式推理模块
├── reasoning_cli.py  # 推理CLI界面
├── action_encoding.py# 动作编码/解码
├── export.py         # 数据与图表导出
├── gui.py            # pygame图形界面
├── cli.py            # 命令行对弈界面
├── tests.py          # 单元测试
└── __main__.py       # 模块主入口
```
