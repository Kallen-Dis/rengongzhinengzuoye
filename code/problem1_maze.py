"""题目一：A* 迷宫寻路，每步代价 1，返回包含首尾的最短路径。

保留 astar_maze(grid, start, goal) 接口，增加启发函数及统计可选参数。
"""
from datasets import MAZE
from search_common import SearchStats, astar_search, manhattan


def astar_maze(grid, start, goal, heuristic="manhattan", stats=None):
    """heuristic='zero' 使用 h=0；无解返回空列表。"""
    if heuristic not in ("zero", "manhattan"):
        raise ValueError("heuristic 必须是 zero 或 manhattan")

    def enter(pos):
        return None if grid[pos[0]][pos[1]] == 'X' else 1

    def estimate(pos):
        return 0 if heuristic == "zero" else manhattan(pos, goal)

    path, _ = astar_search(grid, start, goal, enter, estimate, stats)
    return path


if __name__ == "__main__":
    stats = SearchStats()
    path = astar_maze(MAZE, (0, 0), (3, 4), stats=stats)
    print("最短路径:", path)
    print("路径长度(步数):", len(path) - 1 if path else -1)
    print("扩展节点数:", stats.expanded)
