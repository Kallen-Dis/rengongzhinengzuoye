"""A* 公共核心：题目一与题目二共用，只依赖 Python 标准库。"""
from dataclasses import dataclass
import heapq


@dataclass
class SearchStats:
    visited: int = 0       # 有效出堆次数，包含终点
    expanded: int = 0      # 实际枚举邻居次数，不包含终点
    generated: int = 0     # 入堆记录数，包含起点和改进后的重复坐标
    stale: int = 0         # 跳过的过期记录数
    max_frontier: int = 0  # 堆内记录数峰值，包含尚未清理的旧记录


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def validate_grid(grid, start, goal):
    if not grid or not grid[0]:
        raise ValueError("地图不能为空")
    cols = len(grid[0])
    if any(len(row) != cols for row in grid):
        raise ValueError("地图必须是规则矩形")
    for r, c in (start, goal):
        if not (0 <= r < len(grid) and 0 <= c < cols):
            raise ValueError("起点或终点越界")


def astar_search(grid, start, goal, enter_cost, heuristic, stats=None):
    """返回 (路径, 最小代价)，无解为 ([], -1)。

    enter_cost(pos) 返回非负进入代价，None 表示障碍。
    heuristic(pos) 必须不高估剩余代价。本实验采用一致启发函数。
    不永久封闭坐标；如果发现更小 g，允许重新入堆和扩展。
    """
    validate_grid(grid, start, goal)
    stats = stats if stats is not None else SearchStats()
    if enter_cost(start) is None or enter_cost(goal) is None:
        return [], -1
    g_score = {start: 0}  # 出发不支付起点格费用
    came_from = {}
    # f 相同时按坐标排序，保证路径和节点数可复现。
    open_heap = [(heuristic(start), start, 0)]
    stats.generated = stats.max_frontier = 1
    while open_heap:
        _, current, popped_g = heapq.heappop(open_heap)
        if popped_g != g_score[current]:
            stats.stale += 1
            continue
        stats.visited += 1
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1], popped_g
        stats.expanded += 1
        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (r + dr, c + dc)
            nr, nc = neighbor
            if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0])):
                continue
            cost = enter_cost(neighbor)
            if cost is None:
                continue
            if cost < 0:
                raise ValueError("移动代价不能为负")
            new_g = popped_g + cost
            if new_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = new_g
                came_from[neighbor] = current
                heapq.heappush(open_heap, (new_g + heuristic(neighbor), neighbor, new_g))
                stats.generated += 1
                stats.max_frontier = max(stats.max_frontier, len(open_heap))
    return [], -1
