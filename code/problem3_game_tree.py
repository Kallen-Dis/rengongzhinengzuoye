"""题目三：完整 Minimax 与 Alpha-beta。默认严格从左到右搜索。"""
from dataclasses import dataclass


class TreeNode:
    def __init__(self, value=None, children=None):
        self.value = value
        self.children = children if children is not None else []


@dataclass
class GameStats:
    visited: int = 0
    leaves: int = 0
    cutoffs: int = 0          # 确实跳过了剩余子分支的次数
    pruned_children: int = 0 # 跳过的直接子分支数，不是全部后代数
    cache_hits: int = 0


def sample_tree():
    """依据 PDF 第 10 至 12 页；右侧 MIN 也包含 (6,9) 的 MAX 分支。"""
    def pair(a, b):
        return TreeNode(children=[TreeNode(value=a), TreeNode(value=b)])
    left = TreeNode(children=[pair(3, 5), pair(6, 9)])
    right = TreeNode(children=[pair(6, 9), pair(1, 2)])
    return TreeNode(children=[left, right])


def tree_from_dict(data):
    """兼容 PDF 的 JSON；玩家从根 MAX 开始逐层交替。"""
    if "tree_structure" in data:
        data = data["tree_structure"]
    if data.get("children"):
        return TreeNode(children=[tree_from_dict(x) for x in data["children"]])
    if "value" not in data:
        raise ValueError("叶子节点必须有 value")
    return TreeNode(value=data["value"])


def minimax(node, is_maximizing=True, stats=None, order="left"):
    if order not in ("left", "reverse"):
        raise ValueError("order 必须是 left 或 reverse")
    stats = stats if stats is not None else GameStats()
    stats.visited += 1
    if not node.children:
        if node.value is None:
            raise ValueError("叶子节点缺少分数")
        stats.leaves += 1
        return node.value
    children = node.children if order == "left" else list(reversed(node.children))
    best = -float("inf") if is_maximizing else float("inf")
    for child in children:
        value = minimax(child, not is_maximizing, stats, order)
        best = max(best, value) if is_maximizing else min(best, value)
    return best


def alpha_beta(node, alpha=-float("inf"), beta=float("inf"),
               is_maximizing=True, stats=None, order="left"):
    """MAX 更新 alpha 下界，MIN 更新 beta 上界，相交时剪枝。"""
    if order not in ("left", "reverse"):
        raise ValueError("order 必须是 left 或 reverse")
    stats = stats if stats is not None else GameStats()
    stats.visited += 1
    if not node.children:
        if node.value is None:
            raise ValueError("叶子节点缺少分数")
        stats.leaves += 1
        return node.value
    children = node.children if order == "left" else list(reversed(node.children))
    best = -float("inf") if is_maximizing else float("inf")
    for i, child in enumerate(children):
        value = alpha_beta(child, alpha, beta, not is_maximizing, stats, order)
        if is_maximizing:
            best = max(best, value)
            alpha = max(alpha, best)
        else:
            best = min(best, value)
            beta = min(beta, best)
        if alpha >= beta and i + 1 < len(children):
            stats.cutoffs += 1
            stats.pruned_children += len(children) - i - 1
            break
    return best


if __name__ == "__main__":
    for name, function in [("Minimax", minimax), ("Alpha-beta", alpha_beta)]:
        stats = GameStats()
        print(name, "最佳得分:", function(sample_tree(), stats=stats), "统计:", stats)
