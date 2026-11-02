"""题目四：A* 求移动代价，博弈搜索决定宝藏。每层是选择一次宝藏。"""
from dataclasses import dataclass
from datasets import POINTS, TREASURE_GRID, VALUES
from problem2_weighted import astar_weighted
from problem3_game_tree import GameStats
from search_common import SearchStats


@dataclass(frozen=True)
class GameState:
    """不可变状态，避免不同搜索分支相互污染。位置用关键点名称表示。"""
    pos_a: str = "A"
    pos_b: str = "B"
    remaining: tuple = ("T1", "T2", "T3", "T4")
    score_a: int = 0
    score_b: int = 0
    turn: str = "A"

    @property
    def utility(self):
        return self.score_a - self.score_b


def distance_matrix(grid=TREASURE_GRID, points=POINTS, heuristic="manhattan"):
    """逐方向计算，避免假定一般加权地图的距离对称。"""
    distances, counters = {}, SearchStats()
    for source, start in points.items():
        distances[source] = {}
        for target, goal in points.items():
            if source == target:
                distances[source][target] = 0
                continue
            stats = SearchStats()
            distances[source][target] = astar_weighted(grid, start, goal, heuristic, stats)
            for field in ("visited", "expanded", "generated", "stale"):
                setattr(counters, field, getattr(counters, field) + getattr(stats, field))
            counters.max_frontier = max(counters.max_frontier, stats.max_frontier)
    return distances, counters


def legal_targets(state, distances):
    pos = state.pos_a if state.turn == "A" else state.pos_b
    return [t for t in state.remaining if distances[pos][t] >= 0]


def move(state, target, distances, values=VALUES):
    """只收集指定宝藏；负收益也必须行动，题目没有跳过回合规则。"""
    if target not in state.remaining:
        raise ValueError("宝藏已经被获取或不存在")
    pos = state.pos_a if state.turn == "A" else state.pos_b
    cost = distances[pos][target]
    if cost < 0:
        raise ValueError("目标不可达")
    gain = values[target] - cost
    remaining = tuple(t for t in state.remaining if t != target)
    if state.turn == "A":
        return GameState(target, state.pos_b, remaining,
                         state.score_a + gain, state.score_b, "B")
    return GameState(state.pos_a, target, remaining,
                     state.score_a, state.score_b + gain, "A")


def ordered_targets(state, distances, values, order):
    targets = legal_targets(state, distances)
    if order == "natural":
        return targets
    if order == "reverse":
        return list(reversed(targets))
    pos = state.pos_a if state.turn == "A" else state.pos_b
    if order in ("gain", "low_gain"):
        # 双方都优先看自己的高收益；MIN 增分会使 U 减少。
        direction = -1 if order == "gain" else 1
        return sorted(targets,
                      key=lambda t: (direction * (values[t] - distances[pos][t]), t))
    raise ValueError("未知访问顺序")


def solve_game(distances, depth=4, algorithm="alphabeta", order="natural",
               state=None, values=VALUES, stats=None, use_cache=False):
    """返回 {best_target, best_value}。达到深度或无可行动作时返回当前 U。

    可选精确值缓存仅用于完整 Minimax，不缓存 Alpha-beta 的上下界。
    相同最优值时，保留当前访问顺序中的第一个动作。
    """
    if depth < 0:
        raise ValueError("深度必须非负")
    if algorithm not in ("minimax", "alphabeta"):
        raise ValueError("algorithm 必须是 minimax 或 alphabeta")
    if use_cache and algorithm != "minimax":
        raise ValueError("精确值缓存仅用于完整 Minimax")
    stats = stats if stats is not None else GameStats()
    state = state if state is not None else GameState()
    cache = {}

    def search(current, remaining_depth, alpha, beta):
        stats.visited += 1
        key = (current, remaining_depth)
        if use_cache and key in cache:
            stats.cache_hits += 1
            return cache[key]
        targets = ordered_targets(current, distances, values, order)
        if remaining_depth == 0 or not targets:
            stats.leaves += 1
            result = (None, current.utility)
            if use_cache:
                cache[key] = result
            return result
        maximizing = current.turn == "A"
        best_value = -float("inf") if maximizing else float("inf")
        best_target = None
        for i, target in enumerate(targets):
            child = move(current, target, distances, values)
            _, value = search(child, remaining_depth - 1, alpha, beta)
            improved = value > best_value if maximizing else value < best_value
            if improved:
                best_value, best_target = value, target
            if algorithm == "alphabeta":
                if maximizing:
                    alpha = max(alpha, best_value)
                else:
                    beta = min(beta, best_value)
                if alpha >= beta and i + 1 < len(targets):
                    stats.cutoffs += 1
                    stats.pruned_children += len(targets) - i - 1
                    break
        result = (best_target, best_value)
        if use_cache:
            cache[key] = result
        return result

    target, value = search(state, depth, -float("inf"), float("inf"))
    return {"best_target": target, "best_value": value}


def optimal_trace(distances, depth=4):
    """独立重求每一步的完整 Minimax，仅用于解释，不纳入性能计数。"""
    state, trace = GameState(), []
    for remaining_depth in range(depth, 0, -1):
        choice = solve_game(distances, remaining_depth, "minimax", state=state)
        target = choice["best_target"]
        if target is None:
            break
        pos = state.pos_a if state.turn == "A" else state.pos_b
        cost = distances[pos][target]
        new_state = move(state, target, distances)
        trace.append({"player": state.turn, "from": pos, "target": target,
                      "cost": cost, "gain": VALUES[target] - cost,
                      "score_a": new_state.score_a, "score_b": new_state.score_b,
                      "utility": new_state.utility})
        state = new_state
    return trace


if __name__ == "__main__":
    distances, _ = distance_matrix()
    labels = list(POINTS)
    print("距离矩阵：行是出发点，列是目的地")
    print("    " + " ".join(f"{x:>4}" for x in labels))
    for source in labels:
        print(f"{source:>4}" + " ".join(f"{distances[source][t]:>4}" for t in labels))
    for depth in (2, 4):
        for method in ("minimax", "alphabeta"):
            stats = GameStats()
            print(f"H={depth} {method}:", solve_game(distances, depth, method, stats=stats), stats)
