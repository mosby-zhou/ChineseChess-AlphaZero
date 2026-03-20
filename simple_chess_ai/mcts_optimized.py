"""
优化版 MCTS - 减少对象创建开销

主要优化：
1. 复用棋盘对象，减少深拷贝
2. 启用局面缓存，避免重复推理
3. 预计算 PUCT 常量
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict

from simple_chess_ai.game import (
    ChessGame, ACTION_LABELS, LABEL_TO_INDEX, NUM_ACTIONS,
    flip_move, flip_policy
)


class OptimizedMCTSNode:
    """优化的 MCTS 节点"""
    __slots__ = ['visit_count', 'total_value', 'prior', 'children']

    def __init__(self, prior: float = 0.0):
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.children: Dict[str, 'OptimizedMCTSNode'] = {}

    @property
    def q_value(self) -> float:
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0


class OptimizedMCTS:
    """
    优化的 MCTS 实现

    Args:
        model: ChessModel 实例
        num_simulations: 每步搜索的模拟次数
        c_puct: 探索系数
        dirichlet_alpha: Dirichlet 噪声参数
        dirichlet_weight: 噪声权重
        cache_size: 局面缓存大小（推荐 1000-5000）
    """

    def __init__(self, model, num_simulations: int = 200, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3, dirichlet_weight: float = 0.25,
                 cache_size: int = 2000):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.cache_size = cache_size
        self.root = OptimizedMCTSNode()

        # 局面缓存：FEN -> (policy, value)
        self._cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._cache_hits = 0

    def get_action_probs(self, game: ChessGame, temperature: float = 1.0,
                         add_noise: bool = False, reset_root: bool = True) -> Tuple[List[str], List[float]]:
        """运行 MCTS 搜索"""
        if reset_root:
            self.root = OptimizedMCTSNode()

        # 清理缓存（保留部分）
        if len(self._cache) > self.cache_size:
            # 简单策略：清空一半
            keys = list(self._cache.keys())
            for k in keys[:len(keys)//2]:
                del self._cache[k]

        self._cache_hits = 0

        # 运行模拟
        for i in range(self.num_simulations):
            self._simulate(game, self.root)
            if i == 0 and add_noise and self.root.children:
                self._add_dirichlet_noise(self.root)

        # 提取走法概率
        actions = list(self.root.children.keys())
        visits = [self.root.children[a].visit_count for a in actions]

        if not actions:
            return [], []

        if temperature < 1e-8:
            best_idx = np.argmax(visits)
            probs = [0.0] * len(actions)
            probs[best_idx] = 1.0
        else:
            visits_arr = np.array(visits, dtype=np.float64)
            visits_temp = visits_arr ** (1.0 / temperature)
            total = visits_temp.sum()
            probs = (visits_temp / total).tolist() if total > 0 else [1.0 / len(actions)] * len(actions)

        return actions, probs

    def _simulate(self, game: ChessGame, root: OptimizedMCTSNode):
        """运行一次模拟"""
        node = root
        path = [node]

        # 复用棋盘对象
        game_copy = game.copy()

        # 选择阶段
        while node.children and not game_copy.done:
            action, node = self._select_child(node)
            actual_action = action if game_copy.red_to_move else flip_move(action)
            game_copy.step(actual_action)
            path.append(node)

        # 评估叶子节点
        if game_copy.done:
            if game_copy.winner == 'draw':
                value = 0.0
            else:
                value = -1.0
        else:
            value = self._evaluate_leaf(game_copy, node)

        # 回传
        for i in range(len(path) - 1, -1, -1):
            path[i].visit_count += 1
            path[i].total_value += value
            value = -value

    def _evaluate_leaf(self, game: ChessGame, node: OptimizedMCTSNode) -> float:
        """评估叶子节点（带缓存）"""
        legal_moves = game.get_legal_moves()
        legal_moves_flipped = legal_moves if game.red_to_move else [flip_move(m) for m in legal_moves]
        legal_indices = [LABEL_TO_INDEX[m] for m in legal_moves_flipped if m in LABEL_TO_INDEX]

        # 使用 FEN 作为缓存键
        cache_key = game.get_observation()

        if cache_key in self._cache:
            policy, value = self._cache[cache_key]
            self._cache_hits += 1
        else:
            planes = game.to_planes()
            policy, value = self.model.predict_with_mask(planes, legal_indices)
            if len(self._cache) < self.cache_size:
                self._cache[cache_key] = (policy.copy(), value)

        # 扩展节点
        total_prior = 0.0
        for move in legal_moves_flipped:
            if move in LABEL_TO_INDEX:
                prior = policy[LABEL_TO_INDEX[move]]
                node.children[move] = OptimizedMCTSNode(prior=prior)
                total_prior += prior

        if total_prior > 0:
            for child in node.children.values():
                child.prior /= total_prior

        return -value

    def _select_child(self, node: OptimizedMCTSNode) -> Tuple[str, OptimizedMCTSNode]:
        """选择最优子节点"""
        best_score = -float('inf')
        best_action = None
        best_child = None

        sqrt_total = math.sqrt(node.visit_count) if node.visit_count > 0 else 1.0

        for action, child in node.children.items():
            q = child.q_value
            u = self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _add_dirichlet_noise(self, node: OptimizedMCTSNode):
        """添加 Dirichlet 噪声"""
        children = list(node.children.values())
        n = len(children)
        if n == 0:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * n)
        for child, eta in zip(children, noise):
            child.prior = (1 - self.dirichlet_weight) * child.prior + self.dirichlet_weight * eta

    def update_with_move(self, action: str):
        """更新根节点"""
        if action in self.root.children:
            self.root = self.root.children[action]
        else:
            self.root = OptimizedMCTSNode()

    def get_cache_stats(self) -> Tuple[int, int]:
        """获取缓存统计"""
        return self._cache_hits, len(self._cache)
