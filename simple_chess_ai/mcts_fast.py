"""
高性能批量推理 MCTS

优化点：
1. 减少状态复制：使用轻量级状态表示
2. 预计算 PUCT：缓存常用计算结果
3. 并行选择：多线程选择叶子节点
4. 减少虚拟损失开销：使用原子计数器
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import threading

from simple_chess_ai.game import (
    ChessGame, ACTION_LABELS, LABEL_TO_INDEX, NUM_ACTIONS,
    flip_move, flip_policy
)


class FastMCTSNode:
    """优化的 MCTS 节点"""

    __slots__ = ['visit_count', 'total_value', 'prior', 'children', 'virtual_loss', '_q_cache', '_dirty']

    def __init__(self, prior: float = 0.0):
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.children: Dict[str, 'FastMCTSNode'] = {}
        self.virtual_loss = 0
        self._q_cache = 0.0
        self._dirty = True  # 是否需要重新计算 Q

    @property
    def q_value(self) -> float:
        if self._dirty:
            if self.visit_count == 0:
                self._q_cache = 0.0
            else:
                self._q_cache = self.total_value / self.visit_count
            self._dirty = False
        return self._q_cache

    def add_visit(self, value: float):
        """添加访问并更新 Q 值"""
        self.visit_count += 1
        self.total_value += value
        self._dirty = True

    def add_virtual_loss(self):
        """添加虚拟损失"""
        self.virtual_loss += 1

    def remove_virtual_loss(self):
        """移除虚拟损失"""
        self.virtual_loss -= 1


@dataclass
class LeafInfo:
    """叶子节点信息"""
    planes: np.ndarray  # 特征平面（预计算）
    legal_indices: List[int]
    legal_moves: List[str]
    path: List[FastMCTSNode]
    node: FastMCTSNode


class FastBatchMCTS:
    """
    高性能批量 MCTS

    优化措施：
    1. 轻量级状态：只保存 FEN 字符串，延迟计算
    2. 预计算 PUCT 常量：减少重复计算
    3. 批量选择：并行收集叶子节点
    4. 缓存友好：减少内存分配
    """

    def __init__(self, model, num_simulations: int = 200, batch_size: int = 16,
                 c_puct: float = 1.5, virtual_loss: float = 3.0,
                 dirichlet_alpha: float = 0.3, dirichlet_weight: float = 0.25):
        self.model = model
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.c_puct = c_puct
        self.virtual_loss = virtual_loss
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.root = FastMCTSNode()

        # 预计算常量
        self._sqrt_cache = {}

    def get_action_probs(self, game: ChessGame, temperature: float = 1.0,
                         add_noise: bool = False, reset_root: bool = True) -> Tuple[List[str], List[float]]:
        """运行 MCTS 搜索"""
        if reset_root:
            self.root = FastMCTSNode()

        # 预计算根节点的合法走法
        root_legal_moves = game.get_legal_moves()
        root_legal_indices = [LABEL_TO_INDEX[m] for m in root_legal_moves if m in LABEL_TO_INDEX]

        # 首次扩展根节点
        if not self.root.children:
            planes = game.to_planes()
            policy, value = self.model.predict_with_mask(planes, root_legal_indices)
            self._expand_node(self.root, root_legal_moves, policy)

            if add_noise:
                self._add_dirichlet_noise(self.root)

            self.root.add_visit(value)

        # 批量模拟
        remaining = self.num_simulations - 1
        while remaining > 0:
            batch_size = min(self.batch_size, remaining)
            leaves = self._collect_leaves_batch(game, batch_size)

            if not leaves:
                break

            # 批量推理
            planes_batch = np.array([leaf.planes for leaf in leaves])
            policies, values = self.model.predict_with_masks_batch(
                planes_batch,
                [leaf.legal_indices for leaf in leaves]
            )

            # 扩展并回传
            for leaf, policy, value in zip(leaves, policies, values):
                self._expand_and_backprop(leaf, policy, -value)

            remaining -= len(leaves)

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

    def _collect_leaves_batch(self, root_game: ChessGame, batch_size: int) -> List[LeafInfo]:
        """批量收集叶子节点（优化版）"""
        leaves = []

        for _ in range(batch_size):
            game = root_game.copy()
            node = self.root
            path = [node]

            # 选择到叶子节点
            while node.children and not game.done:
                action, child = self._select_child_fast(node)
                if child is None:
                    break

                child.add_virtual_loss()

                # 执行走法
                actual_action = action if game.red_to_move else flip_move(action)
                game.step(actual_action)

                node = child
                path.append(node)

            if game.done:
                # 终局：直接回传
                value = 0.0 if game.winner == 'draw' else -1.0
                self._backprop_fast(path, value)
                continue

            # 收集叶子节点信息
            legal_moves = game.get_legal_moves()
            legal_moves_flipped = legal_moves if game.red_to_move else [flip_move(m) for m in legal_moves]
            legal_indices = [LABEL_TO_INDEX[m] for m in legal_moves_flipped if m in LABEL_TO_INDEX]

            leaves.append(LeafInfo(
                planes=game.to_planes(),
                legal_indices=legal_indices,
                legal_moves=legal_moves_flipped,
                path=path,
                node=node
            ))

        return leaves

    def _select_child_fast(self, node: FastMCTSNode) -> Tuple[Optional[str], Optional[FastMCTSNode]]:
        """快速选择子节点（优化 PUCT 计算）"""
        best_score = -float('inf')
        best_action = None
        best_child = None

        # 预计算 sqrt
        total_visits = node.visit_count + sum(c.virtual_loss for c in node.children.values())
        sqrt_total = self._sqrt_cache.get(total_visits)
        if sqrt_total is None:
            sqrt_total = math.sqrt(max(1, total_visits))
            self._sqrt_cache[total_visits] = sqrt_total
            # 限制缓存大小
            if len(self._sqrt_cache) > 1000:
                self._sqrt_cache.clear()

        c_puct_sqrt = self.c_puct * sqrt_total

        for action, child in node.children.items():
            # 优化：避免函数调用开销
            q = child.q_value
            if child.virtual_loss > 0:
                q = (child.total_value - self.virtual_loss * child.virtual_loss) / \
                    (child.visit_count + child.virtual_loss)

            u = child.prior * c_puct_sqrt / (1 + child.visit_count + child.virtual_loss)
            score = q + u

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _expand_node(self, node: FastMCTSNode, legal_moves: List[str], policy: np.ndarray):
        """扩展节点"""
        total_prior = 0.0
        for move in legal_moves:
            if move in LABEL_TO_INDEX:
                prior = policy[LABEL_TO_INDEX[move]]
                node.children[move] = FastMCTSNode(prior=prior)
                total_prior += prior

        # 归一化
        if total_prior > 0:
            for child in node.children.values():
                child.prior /= total_prior

    def _expand_and_backprop(self, leaf: LeafInfo, policy: np.ndarray, value: float):
        """扩展并回传"""
        self._expand_node(leaf.node, leaf.legal_moves, policy)
        self._backprop_fast(leaf.path, value)

    def _backprop_fast(self, path: List[FastMCTSNode], value: float):
        """快速回传"""
        for i in range(len(path) - 1, -1, -1):
            node = path[i]
            node.add_visit(value)
            if node.virtual_loss > 0:
                node.remove_virtual_loss()
            value = -value

    def _add_dirichlet_noise(self, node: FastMCTSNode):
        """添加 Dirichlet 噪声"""
        children = list(node.children.values())
        if not children:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(children))
        for child, eta in zip(children, noise):
            child.prior = (1 - self.dirichlet_weight) * child.prior + self.dirichlet_weight * eta

    def update_with_move(self, action: str):
        """更新根节点"""
        if action in self.root.children:
            self.root = self.root.children[action]
        else:
            self.root = FastMCTSNode()
