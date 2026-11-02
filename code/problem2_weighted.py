"""题目二：加权地形最小代价。进入 E 免费，出发不支付起点费用。

题目四的宝藏格需要支付 1，不能当成免费的 E。
"""
from datasets import WEIGHTED
from search_common import SearchStats, astar_search, manhattan, validate_grid


def enter_cost(grid, pos):
    """进入 pos 的代价；None 表示障碍；兼容题目四的特殊格。"""
    cell = grid[pos[0]][pos[1]]
    if cell == 'X':
        return None
    if cell == 'E':
        return 0
    if cell in ('S', 'A', 'B', 'T1', 'T2', 'T3', 'T4'):
        return 1
    value = int(cell)
    if value < 0:
        raise ValueError("地形代价必须非负")
    return value


def make_heuristic(grid, goal, mode="manhattan"):
    """生成安全下界。

    终点免费时，h=max(曼哈顿距离-1,0)*m；m 是其他通行格的最小代价。
    终点收费时，h=曼哈顿距离*m；m 是所有通行格的最小代价。
    若存在其他零代价格，则 m=0，自动退化为 Dijkstra。
    """
    if mode not in ("zero", "manhattan"):
        raise ValueError("heuristic 必须是 zero 或 manhattan")
    free_goal = enter_cost(grid, goal) == 0
    costs = []
    for r, row in enumerate(grid):
        for c, _ in enumerate(row):
            if free_goal and (r, c) == goal:
                continue
            cost = enter_cost(grid, (r, c))
            if cost is not None:
                costs.append(cost)
    minimum = min(costs, default=0)

    def estimate(pos):
        if mode == "zero":
            return 0
        steps = manhattan(pos, goal)
        if free_goal:
            steps = max(steps - 1, 0)
        return steps * minimum

    return estimate


def astar_weighted_path(grid, start, goal, heuristic="manhattan", stats=None):
    """返回 (路径, 代价)，方便报告逐步核算；无解为 ([], -1)。"""
    validate_grid(grid, start, goal)
    estimate = make_heuristic(grid, goal, heuristic)
    return astar_search(grid, start, goal,
                        lambda pos: enter_cost(grid, pos), estimate, stats)


def astar_weighted(grid, start, goal, heuristic="manhattan", stats=None):
    """保留原接口：返回最小总代价，无解返回 -1。"""
    _, cost = astar_weighted_path(grid, start, goal, heuristic, stats)
    return cost


if __name__ == "__main__":
    stats = SearchStats()
    path, cost = astar_weighted_path(WEIGHTED, (0, 0), (4, 4), stats=stats)
    print("最小代价路径:", path)
    print("最小总代价:", cost)
    print("各步进入代价:", [enter_cost(WEIGHTED, p) for p in path[1:]])
    print("扩展节点数:", stats.expanded)
