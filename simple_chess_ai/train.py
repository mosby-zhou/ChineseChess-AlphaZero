"""
训练管线

实现自对弈数据生成和模型训练的完整流程：
1. 自对弈：使用MCTS生成训练数据
2. 训练：用生成的数据训练神经网络
3. 循环：重复以上步骤持续提升

支持训练模式：
- 标准训练（原版AlphaZero策略）
- GRPO训练（Group Relative Policy Optimization）
- FP16混合精度训练
- 多进程并行自对弈

用法：
    python -m simple_chess_ai.train --num_games 100 --num_epochs 10
    python -m simple_chess_ai.train --num_games 100 --use_grpo
    python -m simple_chess_ai.train --num_games 100 --use_fp16
    python -m simple_chess_ai.train --num_games 100 --num_workers 4 --mcts_mode batch
"""

import os
import json
import time
import copy
import random as _random
import argparse
import datetime
import numpy as np
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from simple_chess_ai.game import (
    ChessGame, NUM_ACTIONS, ACTION_LABELS, LABEL_TO_INDEX,
    flip_move, flip_policy, fen_to_planes
)
from simple_chess_ai.model import ChessModel
from simple_chess_ai.mcts import MCTS as StandardMCTS
from simple_chess_ai.mcts_optimized import OptimizedMCTS
from simple_chess_ai.mcts_fast import FastBatchMCTS
from simple_chess_ai.export import (
    init_run_dir, append_self_play_jsonl,
    append_training_csv, append_gating_csv, plot_curves,
    DEFAULT_RUNS_DIR,
)

# 默认模型保存路径
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_model')
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, 'model.pth')
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), 'train_data')


# ── 并行自对弈工作进程 ──────────────────────────────────────────────────
# 模块级全局变量，用于工作进程内缓存模型实例
_worker_model = None


def _init_worker(model_path, num_channels, num_res_blocks, backend):
    """
    工作进程初始化：独立加载模型并使用 torch.jit.trace 优化。

    每个 worker 进程独立加载模型，避免共享 GPU 上下文的问题。
    同时限制每个进程的 CPU 线程数以避免过载。
    """
    global _worker_model
    import torch as _torch
    # 限制每个 worker 的 CPU 线程数，避免进程间竞争
    _torch.set_num_threads(max(1, min(4, _torch.get_num_threads() // 2)))
    _worker_model = ChessModel(
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        backend=backend
    )
    _worker_model.load(model_path)
    # Worker 内也使用 jit.trace 加速推理
    _worker_model.compile()


def _run_game_worker(num_simulations, max_moves, temperature_threshold,
                     mcts_mode, mcts_batch_size):
    """在工作进程中运行一局自对弈，返回 (training_data, winner, move_count, elapsed)。"""
    global _worker_model
    t0 = time.time()
    data, winner, moves = self_play_game(
        _worker_model,
        num_simulations=num_simulations,
        max_moves=max_moves,
        temperature_threshold=temperature_threshold,
        mcts_mode=mcts_mode,
        mcts_batch_size=mcts_batch_size
    )
    elapsed = time.time() - t0
    return data, winner, moves, elapsed


def _log_gpu_usage(prefix=""):
    """打印当前 GPU 使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        device_name = torch.cuda.get_device_name(0)
        print(f"{prefix}GPU [{device_name}] "
              f"VRAM: {allocated:.0f}MB used / {reserved:.0f}MB reserved")


def evaluate_models(model_a, model_b, n_games=20, num_simulations=50, max_moves=200):
    """
    通过对局评测两个模型，返回 model_a 的 score（draw=0.5 计分）。

    对弈时禁用 Dirichlet 噪声，使用温度=0 的确定性走法，
    并交替红黑方以减少先手优势偏差。
    score = (wins_a + 0.5 * draws) / total

    Args:
        model_a: 候选新模型
        model_b: 基准模型
        n_games: 对局数
        num_simulations: MCTS模拟次数
        max_moves: 每局最大步数

    Returns:
        (score, wins_a, wins_b, draws)
        其中 score = (wins_a + 0.5 * draws) / total，draw 计 0.5 分
    """
    wins_a = 0
    wins_b = 0
    draws = 0

    for game_idx in range(n_games):
        # 交替先手以减少先手优势偏差
        if game_idx % 2 == 0:
            red_model, black_model = model_a, model_b
            a_is_red = True
        else:
            red_model, black_model = model_b, model_a
            a_is_red = False

        game = ChessGame()
        game.reset()
        mcts_red = StandardMCTS(red_model, num_simulations=num_simulations)
        mcts_black = StandardMCTS(black_model, num_simulations=num_simulations)

        move_count = 0
        while not game.done and move_count < max_moves:
            mcts = mcts_red if game.red_to_move else mcts_black
            actions, probs = mcts.get_action_probs(game, temperature=0.0, add_noise=False)
            if not actions:
                break

            chosen_action = actions[int(np.argmax(probs))]
            actual_action = chosen_action if game.red_to_move else flip_move(chosen_action)
            game.step(actual_action)
            mcts.update_with_move(chosen_action)
            move_count += 1

        winner = game.winner
        if winner == 'draw' or winner is None:
            draws += 1
        elif (winner == 'red' and a_is_red) or (winner == 'black' and not a_is_red):
            wins_a += 1
        else:
            wins_b += 1

    total = wins_a + wins_b + draws
    score = (wins_a + 0.5 * draws) / total if total > 0 else 0.0
    return score, wins_a, wins_b, draws


def self_play_game(model, num_simulations=100, max_moves=200, temperature_threshold=30,
                   mcts_mode='optimized', mcts_batch_size=32):
    """
    执行一局自对弈

    Args:
        model: ChessModel实例
        num_simulations: MCTS模拟次数
        max_moves: 最大步数（超过判和）
        temperature_threshold: 前N步使用温度1.0探索
        mcts_mode: MCTS模式 ('standard', 'optimized', 'batch')
        mcts_batch_size: 批量MCTS的批大小（仅 mcts_mode='batch' 时生效）

    Returns:
        training_data: [(state_planes, policy_target, value_target), ...]
    """
    game = ChessGame()
    game.reset()

    # 根据 mcts_mode 选择 MCTS 实现
    if mcts_mode == 'batch':
        mcts = FastBatchMCTS(model, num_simulations=num_simulations, batch_size=mcts_batch_size)
    elif mcts_mode == 'optimized':
        mcts = OptimizedMCTS(model, num_simulations=num_simulations, cache_size=2000)
    else:
        mcts = StandardMCTS(model, num_simulations=num_simulations)

    states = []
    policies = []
    players = []  # 记录每步的走子方

    move_count = 0

    while not game.done and move_count < max_moves:
        # 温度控制
        temperature = 1.0 if move_count < temperature_threshold else 0.1

        # MCTS搜索
        actions, probs = mcts.get_action_probs(game, temperature=temperature, add_noise=True)

        if not actions:
            break

        # 记录训练数据
        planes = game.to_planes()
        policy_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for action, prob in zip(actions, probs):
            if action in LABEL_TO_INDEX:
                policy_target[LABEL_TO_INDEX[action]] = prob

        states.append(planes)
        policies.append(policy_target)
        players.append(1 if game.red_to_move else -1)

        # 按概率选择走法
        action_idx = np.random.choice(len(actions), p=probs)
        chosen_action = actions[action_idx]

        # 执行走法
        if not game.red_to_move:
            actual_action = flip_move(chosen_action)
        else:
            actual_action = chosen_action
        game.step(actual_action)
        mcts.update_with_move(chosen_action)

        move_count += 1

    # 确定胜负
    if game.winner == 'red':
        winner = 1
    elif game.winner == 'black':
        winner = -1
    else:
        winner = 0

    # 生成训练数据
    training_data = []
    for state, policy, player in zip(states, policies, players):
        value = winner * player  # 从该玩家视角的评估值
        training_data.append((state, policy, value))

    return training_data, game.winner, move_count


def multi_game_self_play(model, num_games, num_simulations=200, max_moves=200,
                         temperature_threshold=30, mcts_batch_size=32):
    """
    多局并行自对弈，跨局批量推理。

    与逐局串行不同，本函数同时推进多局游戏的 MCTS 搜索，
    将所有活跃游戏的叶子节点合并为一个大 batch 进行 GPU 推理。

    例如 4 局 × batch_size=32 = 128 个位置一次性推理，大幅提升 GPU 利用率。

    Returns:
        list of (training_data, winner, move_count)
    """
    games = []
    mcts_list = []
    game_states = [[] for _ in range(num_games)]
    game_policies = [[] for _ in range(num_games)]
    game_players = [[] for _ in range(num_games)]
    move_counts = [0] * num_games
    active = set(range(num_games))
    completed = {}

    for i in range(num_games):
        game = ChessGame()
        game.reset()
        games.append(game)
        mcts = FastBatchMCTS(model, num_simulations=num_simulations,
                             batch_size=mcts_batch_size)
        mcts_list.append(mcts)

    total_gpu_time = 0.0
    total_wall_time = time.time()

    while active:
        # ── Phase 1: 批量扩展根节点 ────────────────────────────────────
        root_data = []
        root_game_idx = []
        for gi in list(active):
            game = games[gi]
            mcts = mcts_list[gi]
            if game.done:
                continue
            info = mcts.prepare_root(game, add_noise=(move_counts[gi] < temperature_threshold))
            if info is not None:
                root_data.append(info)
                root_game_idx.append(gi)

        if root_data:
            planes = np.array([d[0] for d in root_data])
            legal_indices_list = [d[1] for d in root_data]
            t_gpu0 = time.time()
            policies, values = model.predict_with_masks_batch(planes, legal_indices_list)
            total_gpu_time += time.time() - t_gpu0
            for i, gi in enumerate(root_game_idx):
                mcts_list[gi].apply_root_expansion(
                    root_data[i][2], policies[i], values[i],
                    add_noise=root_data[i][3]
                )

        # ── Phase 2: 跨局批量 MCTS 模拟 ────────────────────────────────
        needs_more = set()
        for gi in list(active):
            game = games[gi]
            mcts = mcts_list[gi]
            if game.done:
                continue
            if mcts.simulations_done < num_simulations:
                needs_more.add(gi)

        while needs_more:
            all_leaves = []
            leaf_game_map = []

            for gi in list(needs_more):
                game = games[gi]
                mcts = mcts_list[gi]
                remaining = num_simulations - mcts.simulations_done
                batch = min(mcts_batch_size, max(1, remaining))
                leaves = mcts.collect_batch(game, batch)
                for leaf in leaves:
                    all_leaves.append(leaf)
                    leaf_game_map.append(gi)
                if mcts.simulations_done >= num_simulations:
                    needs_more.discard(gi)

            if not all_leaves:
                break

            # 单次 GPU 推理（跨所有游戏）
            planes_batch = np.array([l.planes for l in all_leaves])
            t_gpu0 = time.time()
            policies, values = model.predict_with_masks_batch(
                planes_batch,
                [l.legal_indices for l in all_leaves]
            )
            total_gpu_time += time.time() - t_gpu0

            # 分发结果给各局 MCTS
            for i, leaf in enumerate(all_leaves):
                gi = leaf_game_map[i]
                mcts_list[gi].apply_batch_results([leaf], [policies[i]], [values[i]])

        # ── Phase 3: 提取走法并推进游戏 ────────────────────────────────
        new_active = set()
        for gi in list(active):
            game = games[gi]
            mcts = mcts_list[gi]

            if game.done or move_counts[gi] >= max_moves:
                winner = game.winner
                w = 1 if winner == 'red' else (-1 if winner == 'black' else 0)
                training_data = []
                for state, policy, player in zip(game_states[gi], game_policies[gi], game_players[gi]):
                    training_data.append((state, policy, w * player))
                completed[gi] = (training_data, winner, move_counts[gi])
                continue

            # 温度控制
            temperature = 1.0 if move_counts[gi] < temperature_threshold else 0.1

            # 从 MCTS 树中提取走法概率
            actions, probs = mcts.extract_action_probs(temperature=temperature)

            if not actions:
                winner = game.winner
                w = 1 if winner == 'red' else (-1 if winner == 'black' else 0)
                training_data = []
                for state, policy, player in zip(game_states[gi], game_policies[gi], game_players[gi]):
                    training_data.append((state, policy, w * player))
                completed[gi] = (training_data, winner, move_counts[gi])
                continue

            # 记录训练数据
            planes = game.to_planes()
            policy_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
            for action, prob in zip(actions, probs):
                if action in LABEL_TO_INDEX:
                    policy_target[LABEL_TO_INDEX[action]] = prob

            game_states[gi].append(planes)
            game_policies[gi].append(policy_target)
            game_players[gi].append(1 if game.red_to_move else -1)

            # 选择走法
            action_idx = np.random.choice(len(actions), p=probs)
            chosen_action = actions[action_idx]

            # 执行
            actual_action = chosen_action if game.red_to_move else flip_move(chosen_action)
            game.step(actual_action)
            mcts.update_with_move(chosen_action)
            move_counts[gi] += 1

            if game.done or move_counts[gi] >= max_moves:
                winner = game.winner
                w = 1 if winner == 'red' else (-1 if winner == 'black' else 0)
                training_data = []
                for state, policy, player in zip(game_states[gi], game_policies[gi], game_players[gi]):
                    training_data.append((state, policy, w * player))
                completed[gi] = (training_data, winner, move_counts[gi])
            else:
                new_active.add(gi)

        active = new_active

    wall_elapsed = time.time() - total_wall_time
    gpu_pct = total_gpu_time / max(wall_elapsed, 1e-6) * 100
    print(f"  [Multi-Game] {num_games} 局完成, "
          f"GPU推理占比: {gpu_pct:.1f}% "
          f"(GPU: {total_gpu_time:.2f}s / 总: {wall_elapsed:.1f}s)")

    return [completed[i] for i in range(num_games)]


def _log_gpu_detailed(prefix=""):
    """打印详细 GPU 使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        device_name = torch.cuda.get_device_name(0)
        util = 0.0
        try:
            util = torch.cuda.utilization(0)
        except Exception:
            pass
        print(f"{prefix}GPU [{device_name}] "
              f"Util: {util:.0f}% | "
              f"VRAM: {allocated:.0f}MB / {reserved:.0f}MB")


def train_model(model, training_data, batch_size=256, epochs=5, lr=0.001,
                use_fp16=False):
    """
    用训练数据训练模型

    Args:
        model: ChessModel实例
        training_data: [(planes, policy, value), ...]
        batch_size: 批大小
        epochs: 训练轮数
        lr: 学习率
        use_fp16: 是否使用FP16混合精度训练

    Returns:
        avg_loss: 平均损失
    """
    if not training_data:
        return 0.0

    # 准备数据
    states = np.array([d[0] for d in training_data])
    policies = np.array([d[1] for d in training_data])
    values = np.array([d[2] for d in training_data], dtype=np.float32)

    use_cuda = model.device.type == 'cuda'
    dataset = TensorDataset(
        torch.FloatTensor(states),
        torch.FloatTensor(policies),
        torch.FloatTensor(values)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            pin_memory=use_cuda, num_workers=0)

    # 设置训练模式
    model.model.train()
    optimizer = optim.Adam(model.model.parameters(), lr=lr, weight_decay=1e-4)

    # FP16 混合精度
    amp_enabled = use_fp16 and use_cuda
    scaler = torch.amp.GradScaler('cuda') if amp_enabled else None

    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        for batch_states, batch_policies, batch_values in dataloader:
            batch_states = batch_states.to(model.device, non_blocking=True)
            batch_policies = batch_policies.to(model.device, non_blocking=True)
            batch_values = batch_values.to(model.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=amp_enabled):
                # 前向传播；模型输出 logits（策略头不含 softmax）
                pred_logits, pred_values = model.model(batch_states)

                # 策略损失：交叉熵（用 log_softmax 数值更稳定，避免 log(softmax+eps)）
                policy_loss = -torch.mean(
                    torch.sum(batch_policies * torch.nn.functional.log_softmax(pred_logits, dim=1), dim=1)
                )
                # 价值损失：均方误差
                value_loss = torch.mean((batch_values - pred_values.squeeze()) ** 2)
                # 总损失
                loss = policy_loss + value_loss

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    model.model.eval()
    return total_loss / max(num_batches, 1)


def _save_training_config(model_path, config):
    """将训练配置写入 saved_model/config.json，方便复现与追溯。"""
    import datetime
    model_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else '.'
    os.makedirs(model_dir, exist_ok=True)
    config['_start_time'] = datetime.datetime.now().isoformat(timespec='seconds')
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"训练配置已保存到: {config_path}")


def _do_grpo_train(grpo_trainer, model, data_buffer, batch_size):
    """执行一步 GRPO 训练，返回 (avg_loss, metrics_dict) 或 (0.0, None)"""
    if len(data_buffer) < batch_size:
        return 0.0, None

    batch_data = _random.sample(list(data_buffer), min(batch_size, len(data_buffer)))
    b_states = np.array([d[0] for d in batch_data])
    b_policies = np.array([d[1] for d in batch_data])
    b_values = np.array([d[2] for d in batch_data], dtype=np.float32)

    # 构建合法走法掩码（用策略分布的非零项作为代理）
    b_masks = (b_policies > 0).astype(np.float32)
    # 取策略分布中概率最大的动作
    b_actions = np.argmax(b_policies, axis=1)

    grpo_metrics = grpo_trainer.train_step_from_trajectory(
        b_states, b_masks, b_actions, b_policies, b_values
    )
    return grpo_metrics.get('loss', 0.0), grpo_metrics


def _do_standard_train(model, data_buffer, batch_size, num_epochs, lr, use_fp16):
    """执行一步标准训练，返回 avg_loss"""
    train_data = list(data_buffer)
    return train_model(
        model, train_data, batch_size=batch_size,
        epochs=num_epochs, lr=lr, use_fp16=use_fp16
    )


def _process_completed_game(game_idx, num_games, data, winner, moves, elapsed,
                            data_buffer, stats, model, batch_size, num_epochs,
                            lr, use_fp16, use_grpo, grpo_trainer,
                            run_dir, save_interval, last_model_path,
                            gating_interval, gating_games, gating_winrate,
                            best_model_state, model_path,
                            num_simulations=100, max_moves=200):
    """处理一局完成的游戏：更新统计、训练、保存、gating。"""
    data_buffer.extend(data)

    # 统计
    if winner == 'red':
        stats['red_wins'] += 1
    elif winner == 'black':
        stats['black_wins'] += 1
    else:
        stats['draws'] += 1

    winner_display = {'red': '红方', 'black': '黑方', 'draw': '和棋'}.get(winner, '和棋')
    print(f"[第 {game_idx}/{num_games} 局] "
          f"胜方: {winner_display}, "
          f"步数: {moves}, "
          f"新增数据: {len(data)}, "
          f"缓冲区: {len(data_buffer)}, "
          f"耗时: {elapsed:.1f}s")

    # ── 自对弈记录落盘（JSONL）──────────────────────────────────────────
    append_self_play_jsonl(run_dir, {
        'game_idx': game_idx,
        'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
        'winner': winner or 'draw',
        'num_moves': moves,
        'num_samples': len(data),
        'elapsed_s': round(elapsed, 2),
    })

    # 训练（每局都训练，但数据足够时才有效）
    avg_loss = 0.0
    if len(data_buffer) >= batch_size:
        if use_grpo and grpo_trainer is not None:
            avg_loss, grpo_metrics = _do_grpo_train(
                grpo_trainer, model, data_buffer, batch_size
            )
            if grpo_metrics:
                print(f"  GRPO训练完成，损失: {grpo_metrics['loss']:.4f}, "
                      f"策略损失: {grpo_metrics['policy_loss']:.4f}")
        else:
            avg_loss = _do_standard_train(
                model, data_buffer, batch_size, num_epochs, lr, use_fp16
            )
            print(f"  训练完成，平均损失: {avg_loss:.4f}")

    # ── 训练指标落盘（CSV）──────────────────────────────────────────────
    if len(data_buffer) >= batch_size:
        append_training_csv(run_dir, {
            'game_idx': game_idx,
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
            'loss': round(avg_loss, 6),
            'buffer_size': len(data_buffer),
            'elapsed_s': round(elapsed, 2),
        })

    # 定期保存模型（候选模型保存为 last.pth，不覆盖 best）
    if game_idx % save_interval == 0:
        model.save(last_model_path)
        print(f"  候选模型已保存到: {last_model_path}")

    # Gating：定期评测新模型 vs 基准模型
    if gating_interval > 0 and game_idx % gating_interval == 0:
        print(f"  [Gating] 开始评测 (第 {game_idx} 局后)...")
        ref_model = ChessModel(
            num_channels=model.num_channels,
            num_res_blocks=model.num_res_blocks,
            backend=model.backend
        )
        ref_model.build()
        ref_model.model.load_state_dict(copy.deepcopy(best_model_state[0]))
        ref_model.model.eval()

        score, wins, losses, draws_g = evaluate_models(
            model, ref_model,
            n_games=gating_games,
            num_simulations=max(num_simulations // 2, 20),
            max_moves=max_moves
        )
        print(f"  [Gating] 胜 {wins} / 负 {losses} / 和 {draws_g}, "
              f"score: {score:.0%}, 阈值: {gating_winrate:.0%}")
        if score > gating_winrate:
            print(f"  [Gating] [PASS] 新模型被接受，更新 best 模型")
            best_model_state[0] = copy.deepcopy(model.model.state_dict())
            model.save(model_path)
            print(f"  [Gating] best 模型已保存到: {model_path}")
            gating_accepted = True
        else:
            print(f"  [Gating] [FAIL] 新模型未超过阈值，best 模型不变，训练继续")
            gating_accepted = False

        # ── Gating 评测结果落盘（CSV）────────────────────────────────────
        append_gating_csv(run_dir, {
            'game_idx': game_idx,
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
            'wins_a': wins,
            'wins_b': losses,
            'draws': draws_g,
            'score': round(score, 4),
            'gating_winrate': gating_winrate,
            'accepted': gating_accepted,
        })


def run_training(num_games=50, num_simulations=100, num_epochs=5,
                 batch_size=256, lr=0.001, max_moves=200,
                 buffer_size=10000, model_path=None, save_interval=10,
                 use_grpo=False, grpo_group_size=8, use_fp16=False,
                 gating_interval=20, gating_games=20, gating_winrate=0.55,
                 seed=None, deterministic=False, runs_dir=None, quick=False,
                 mcts_mode='optimized', mcts_batch_size=32, backend='cnn',
                 num_workers=1):
    """
    运行完整的训练流程

    Args:
        num_games: 总自对弈局数
        num_simulations: MCTS模拟次数
        num_epochs: 每次训练的轮数
        batch_size: 批大小
        lr: 学习率
        max_moves: 每局最大步数
        buffer_size: 训练数据缓冲区大小
        model_path: 模型保存路径
        save_interval: 每隔多少局保存一次模型
        use_grpo: 是否使用GRPO训练
        grpo_group_size: GRPO组采样大小
        use_fp16: 是否使用FP16混合精度训练
        gating_interval: 每隔多少局进行一次 gating 评测（0 表示禁用）
        gating_games: gating 评测对局数
        gating_winrate: gating 接受阈值（新模型胜率需超过此值）
        seed: 随机种子（None 表示不固定）
        deterministic: 是否启用 cuDNN 确定性模式（可能降低训练速度）
        runs_dir: 日志与数据导出根目录（默认 simple_chess_ai/runs/，None 表示使用默认）
        mcts_mode: MCTS模式
        mcts_batch_size: 批量MCTS批大小
        num_workers: 并行自对弈工作进程数（1=串行，>1=并行）
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    # 快速模式：减少训练量以快速验证流程
    if quick:
        num_games = 1
        num_simulations = 10
        num_epochs = 1
        batch_size = 16
        gating_interval = 0
        max_moves = 50

    # best 模型路径 = model_path；周期性保存使用同目录下的 last.pth
    model_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else '.'
    last_model_path = os.path.join(model_dir, 'last.pth')

    # ── 可复现性：设置随机种子 ──────────────────────────────────────────────
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            # 启用 cuDNN 确定性选项；注意可能降低训练速度
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    elif torch.cuda.is_available():
        # 非确定性模式下启用 cuDNN 自动调优，加速卷积运算
        torch.backends.cudnn.benchmark = True

    # 初始化模型
    model = ChessModel(num_channels=128, num_res_blocks=4, backend=backend)
    if os.path.exists(model_path):
        print(f"加载已有模型: {model_path}")
        model.load(model_path)
    else:
        print("创建新模型")
        model.build()
        # best 模型文件不存在时，用随机初始化的模型作为初始 best 并保存
        os.makedirs(model_dir, exist_ok=True)
        model.save(model_path)
        print(f"初始化 best 模型已保存到: {model_path}")

    # 使用 torch.compile() 加速推理/训练 (PyTorch 2.0+)
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            model.compile()
        except Exception as e:
            print(f"torch.compile() 跳过: {e}")

    # 保存本次训练的配置（供复现/追溯）
    training_config = dict(
        num_games=num_games, num_simulations=num_simulations,
        num_epochs=num_epochs, batch_size=batch_size, lr=lr,
        max_moves=max_moves, buffer_size=buffer_size,
        save_interval=save_interval, use_grpo=use_grpo,
        grpo_group_size=grpo_group_size, use_fp16=use_fp16,
        gating_interval=gating_interval, gating_games=gating_games,
        gating_winrate=gating_winrate, seed=seed,
        deterministic=deterministic, num_workers=num_workers,
        mcts_mode=mcts_mode, mcts_batch_size=mcts_batch_size,
    )
    _save_training_config(model_path, training_config)

    # ── 初始化数据导出目录 ────────────────────────────────────────────────────
    run_dir = init_run_dir(runs_dir=runs_dir, config=training_config)
    print(f"数据导出目录: {run_dir}")

    # 记录 best 模型权重（仅用于 gating 对比，不回滚训练）
    # 使用列表包装以便在 _process_completed_game 中可变
    best_model_state = [copy.deepcopy(model.model.state_dict())]

    # GRPO 训练器
    grpo_trainer = None
    if use_grpo:
        from simple_chess_ai.grpo import GRPOTrainer
        grpo_trainer = GRPOTrainer(
            model, group_size=grpo_group_size,
            lr=lr, use_fp16=use_fp16
        )

    # 训练数据缓冲区
    data_buffer = deque(maxlen=buffer_size)

    stats = {'red_wins': 0, 'black_wins': 0, 'draws': 0}

    training_mode = "GRPO" if use_grpo else "Standard"
    fp16_str = " + FP16" if use_fp16 else ""
    parallel_str = f" | {num_workers} Workers" if num_workers > 1 else ""

    print(f"\n{'='*60}")
    print(f"开始训练 ({training_mode}{fp16_str}{parallel_str})")
    print(f"自对弈局数: {num_games}")
    print(f"MCTS模拟次数: {num_simulations}")
    if mcts_mode == 'batch':
        print(f"MCTS批大小: {mcts_batch_size}")
    if use_grpo:
        print(f"GRPO组大小: {grpo_group_size}")
    if gating_interval > 0:
        print(f"Gating: 每 {gating_interval} 局评测 {gating_games} 局, 阈值 {gating_winrate:.0%}")
    print(f"模型保存路径: {model_path}")
    _log_gpu_usage("初始 ")
    print(f"{'='*60}\n")

    wall_start = time.time()

    # ── 训练主循环 ─────────────────────────────────────────────────────────
    if num_workers > 1:
        # ════════════════════════════════════════════════════════════════
        # 并行自对弈模式：使用 ProcessPoolExecutor 多进程同时生成对局数据
        # 每个 worker 独立运行 MCTS + GPU 推理，避免 Python GIL 串行化
        # ════════════════════════════════════════════════════════════════
        # 保存最新模型供 worker 加载
        model.save(model_path)

        executor = ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(model_path, model.num_channels, model.num_res_blocks, model.backend)
        )

        try:
            pending = {}  # future -> game_idx
            game_idx = 0
            completed_count = 0

            # 提交初始批次（2x workers 保持流水线满载）
            submit_count = min(num_workers * 2, num_games)
            for _ in range(submit_count):
                game_idx += 1
                f = executor.submit(
                    _run_game_worker,
                    num_simulations, max_moves, 30,
                    mcts_mode, mcts_batch_size
                )
                pending[f] = game_idx

            print(f"  已提交 {submit_count} 局到 {num_workers} 个工作进程 (MCTS批大小={mcts_batch_size})...")

            while pending:
                # 等待至少一个游戏完成（30s 超时打印进度）
                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED,
                               timeout=30.0)

                if not done:
                    elapsed_total = time.time() - wall_start
                    running = len(pending)
                    print(f"  [进度] {completed_count}/{num_games} 局已完成, "
                          f"{running} 局运行中, 总耗时 {elapsed_total:.0f}s")
                    _log_gpu_detailed("  ")
                    continue

                for f in done:
                    gid = pending.pop(f)
                    completed_count += 1

                    data, winner, moves, elapsed = f.result()

                    _process_completed_game(
                        gid, num_games, data, winner, moves, elapsed,
                        data_buffer, stats, model, batch_size, num_epochs,
                        lr, use_fp16, use_grpo, grpo_trainer,
                        run_dir, save_interval, last_model_path,
                        gating_interval, gating_games, gating_winrate,
                        best_model_state, model_path,
                        num_simulations=num_simulations, max_moves=max_moves
                    )

                    # 提交下一局
                    if game_idx < num_games:
                        game_idx += 1
                        new_f = executor.submit(
                            _run_game_worker,
                            num_simulations, max_moves, 30,
                            mcts_mode, mcts_batch_size
                        )
                        pending[new_f] = game_idx

                # 周期性打印 GPU 使用情况
                if completed_count % (num_workers * 2) == 0 and completed_count > 0:
                    _log_gpu_detailed("  ")

        finally:
            executor.shutdown(wait=True)

    else:
        # ════════════════════════════════════════════════════════════════
        # 串行模式（原始逻辑）
        # ════════════════════════════════════════════════════════════════
        for game_idx in range(1, num_games + 1):
            start_time = time.time()

            # 自对弈
            data, winner, moves = self_play_game(
                model, num_simulations=num_simulations, max_moves=max_moves,
                mcts_mode=mcts_mode, mcts_batch_size=mcts_batch_size
            )
            data_buffer.extend(data)

            # 统计
            if winner == 'red':
                stats['red_wins'] += 1
            elif winner == 'black':
                stats['black_wins'] += 1
            else:
                stats['draws'] += 1

            elapsed = time.time() - start_time
            winner_display = {'red': '红方', 'black': '黑方', 'draw': '和棋'}.get(winner, '和棋')
            print(f"[第 {game_idx}/{num_games} 局] "
                  f"胜方: {winner_display}, "
                  f"步数: {moves}, "
                  f"新增数据: {len(data)}, "
                  f"缓冲区: {len(data_buffer)}, "
                  f"耗时: {elapsed:.1f}s")

            # ── 自对弈记录落盘（JSONL）──────────────────────────────────────
            append_self_play_jsonl(run_dir, {
                'game_idx': game_idx,
                'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                'winner': winner or 'draw',
                'num_moves': moves,
                'num_samples': len(data),
                'elapsed_s': round(elapsed, 2),
            })

            # 训练（每局都训练，但数据足够时才有效）
            avg_loss = 0.0
            if len(data_buffer) >= batch_size:
                if use_grpo and grpo_trainer is not None:
                    avg_loss, grpo_metrics = _do_grpo_train(
                        grpo_trainer, model, data_buffer, batch_size
                    )
                    if grpo_metrics:
                        print(f"  GRPO训练完成，损失: {grpo_metrics['loss']:.4f}, "
                              f"策略损失: {grpo_metrics['policy_loss']:.4f}")
                else:
                    avg_loss = _do_standard_train(
                        model, data_buffer, batch_size, num_epochs, lr, use_fp16
                    )
                    print(f"  训练完成，平均损失: {avg_loss:.4f}")

            # ── 训练指标落盘（CSV）──────────────────────────────────────────
            if len(data_buffer) >= batch_size:
                append_training_csv(run_dir, {
                    'game_idx': game_idx,
                    'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                    'loss': round(avg_loss, 6),
                    'buffer_size': len(data_buffer),
                    'elapsed_s': round(elapsed, 2),
                })

            # 定期保存模型（候选模型保存为 last.pth，不覆盖 best）
            if game_idx % save_interval == 0:
                model.save(last_model_path)
                print(f"  候选模型已保存到: {last_model_path}")

            # Gating：定期评测新模型 vs 基准模型
            if gating_interval > 0 and game_idx % gating_interval == 0:
                print(f"  [Gating] 开始评测 (第 {game_idx} 局后)...")
                ref_model = ChessModel(
                    num_channels=model.num_channels,
                    num_res_blocks=model.num_res_blocks,
                    backend=model.backend
                )
                ref_model.build()
                ref_model.model.load_state_dict(copy.deepcopy(best_model_state[0]))
                ref_model.model.eval()

                score, wins, losses, draws_g = evaluate_models(
                    model, ref_model,
                    n_games=gating_games,
                    num_simulations=max(num_simulations // 2, 20),
                    max_moves=max_moves
                )
                print(f"  [Gating] 胜 {wins} / 负 {losses} / 和 {draws_g}, "
                      f"score: {score:.0%}, 阈值: {gating_winrate:.0%}")
                if score > gating_winrate:
                    print(f"  [Gating] [PASS] 新模型被接受，更新 best 模型")
                    best_model_state[0] = copy.deepcopy(model.model.state_dict())
                    model.save(model_path)
                    print(f"  [Gating] best 模型已保存到: {model_path}")
                    gating_accepted = True
                else:
                    print(f"  [Gating] [FAIL] 新模型未超过阈值，best 模型不变，训练继续")
                    gating_accepted = False

                # ── Gating 评测结果落盘（CSV）────────────────────────────────
                append_gating_csv(run_dir, {
                    'game_idx': game_idx,
                    'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
                    'wins_a': wins,
                    'wins_b': losses,
                    'draws': draws_g,
                    'score': round(score, 4),
                    'gating_winrate': gating_winrate,
                    'accepted': gating_accepted,
                })

    # ── 训练结束 ──────────────────────────────────────────────────────────
    wall_elapsed = time.time() - wall_start

    # 最终保存候选模型（best 已在 gating 通过时实时更新到 model_path）
    model.save(last_model_path)
    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"红方胜: {stats['red_wins']}, "
          f"黑方胜: {stats['black_wins']}, "
          f"和棋: {stats['draws']}")
    print(f"总耗时: {wall_elapsed:.1f}s, "
          f"每局平均: {wall_elapsed / max(num_games, 1):.1f}s")
    print(f"最终候选模型已保存到: {last_model_path}")
    print(f"best 模型路径: {model_path}")
    print(f"数据导出目录: {run_dir}")
    _log_gpu_usage("最终 ")

    # ── 自动生成图表 ─────────────────────────────────────────────────────────
    print("正在生成训练曲线图...")
    plot_curves(run_dir)

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='简化中国象棋AI - 训练')
    parser.add_argument('--num_games', type=int, default=50,
                        help='自对弈局数 (默认: 50)')
    parser.add_argument('--num_simulations', type=int, default=100,
                        help='每步MCTS模拟次数 (默认: 100)')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='每次训练轮数 (默认: 5)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='训练批大小 (默认: 256)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率 (默认: 0.001)')
    parser.add_argument('--max_moves', type=int, default=200,
                        help='每局最大步数 (默认: 200)')
    parser.add_argument('--buffer_size', type=int, default=10000,
                        help='训练数据缓冲区大小 (默认: 10000)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型保存路径')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='每隔多少局保存模型 (默认: 10)')
    parser.add_argument('--use_grpo', action='store_true',
                        help='使用GRPO训练模式')
    parser.add_argument('--grpo_group_size', type=int, default=8,
                        help='GRPO组采样大小 (默认: 8)')
    parser.add_argument('--use_fp16', action='store_true',
                        help='使用FP16混合精度训练')
    parser.add_argument('--gating_interval', type=int, default=20,
                        help='每隔多少局进行 gating 评测，0 表示禁用 (默认: 20)')
    parser.add_argument('--gating_games', type=int, default=20,
                        help='gating 评测对局数 (默认: 20)')
    parser.add_argument('--gating_winrate', type=float, default=0.55,
                        help='gating 接受阈值，新模型胜率需超过此值 (默认: 0.55)')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子，设置后可复现数据生成序列 (默认: None)')
    parser.add_argument('--deterministic', action='store_true',
                        help='开启 cuDNN 确定性模式（配合 --seed 使用，可能降低训练速度）')
    parser.add_argument('--runs_dir', type=str, default=None,
                        help=f'数据与日志导出根目录 (默认: simple_chess_ai/runs/)')
    parser.add_argument('--mcts_mode', type=str, default='optimized',
                        choices=['standard', 'optimized', 'batch'],
                        help='MCTS模式: standard(原始), optimized(缓存优化), batch(批量推理最快) (默认: optimized)')
    parser.add_argument('--mcts_batch_size', type=int, default=32,
                        help='批量MCTS的批大小，仅mcts_mode=batch时生效 (默认: 32)')
    parser.add_argument('--backend', type=str, default='cnn',
                        choices=['cnn', 'gnn'],
                        help='网络架构: cnn(卷积网络,速度快) 或 gnn(图网络,理论上限高) (默认: cnn)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='并行自对弈工作进程数，1=串行 (默认: 1)')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式：1局自对弈+1次参数更新，用于验证流程')

    args = parser.parse_args()
    run_training(**vars(args))


if __name__ == '__main__':
    main()
