"""一键运行四题、对照实验与计时，保存 JSON / CSV / 纯文本结果。"""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
from datasets import MAZE, WEIGHTED, POINTS, TREASURE_GRID, VALUES
from problem1_maze import astar_maze
from problem2_weighted import astar_weighted_path, enter_cost
from problem3_game_tree import GameStats, sample_tree, minimax, alpha_beta
from problem4_treasure import (GameState, distance_matrix, solve_game,
                               optimal_trace, move)
from search_common import SearchStats


def timed(call, repeats):
    for _ in range(3):
        call()
    durations = []
    for _ in range(repeats):
        before = time.perf_counter_ns()
        call()
        durations.append((time.perf_counter_ns() - before) / 1_000_000)
    return round(statistics.median(durations), 6)


def extra_mazes():
    size = 25
    open_grid = [['.' for _ in range(size)] for _ in range(size)]
    wall = [row[:] for row in open_grid]
    for r in range(size - 1):
        wall[r][12] = 'X'
    rng = random.Random(2026)
    random_grid = [['X' if rng.random() < 0.22 else '.' for _ in range(size)]
                   for _ in range(size)]
    # 保证至少有一条沿顶行、右列的路线，并保留其他随机通道。
    for i in range(size):
        random_grid[0][i] = random_grid[i][size - 1] = '.'
    return [("open_25", open_grid, (12, 0), (12, 24)),
            ("wall_gap_25", wall, (0, 0), (0, 24)),
            ("random_25", random_grid, (0, 0), (24, 24))]


def collect(repeats):
    rows = []
    data = {"metadata": {"python": platform.python_version(),
                         "platform": platform.platform(), "repeats": repeats,
                         "warmups": 3, "time_unit": "ms",
                         "time_statistic": "median", "random_seed": 2026,
                         "tie_break": "A* 按 (f, 坐标, 入堆g)，博弈保留首个最优动作"}}
    for name, grid, start, goal in [("sample_maze", MAZE, (0, 0), (3, 4))] + extra_mazes():
        for heuristic in ("zero", "manhattan"):
            def call(grid=grid, start=start, goal=goal, heuristic=heuristic):
                stats = SearchStats()
                path = astar_maze(grid, start, goal, heuristic, stats)
                return path, stats
            path, stats = call()
            rows.append({"problem": 1, "case": name, "algorithm": "astar",
                         "heuristic": heuristic, "value": len(path) - 1 if path else -1,
                         **asdict(stats), "median_ms": timed(call, repeats)})
            if name == "sample_maze" and heuristic == "manhattan":
                data["problem1"] = {"path": path, "steps": len(path) - 1}

    weighted_cases = [
        ("sample_weighted", WEIGHTED, (0, 0), (4, 4)),
        ("free_adjacent", [['S', 'E']], (0, 0), (0, 1)),
        ("longer_but_cheaper", [['S', '9', 'E'], ['1', '1', '1']], (0, 0), (0, 2)),
        ("unreachable", [['S', 'X', 'E']], (0, 0), (0, 2)),
        ("zero_terrain", [['S', '0', '0', 'E']], (0, 0), (0, 3)),
    ]
    for name, grid, start, goal in weighted_cases:
        for heuristic in ("zero", "manhattan"):
            def call(grid=grid, start=start, goal=goal, heuristic=heuristic):
                stats = SearchStats()
                path, cost = astar_weighted_path(grid, start, goal, heuristic, stats)
                return path, cost, stats
            path, cost, stats = call()
            rows.append({"problem": 2, "case": name, "algorithm": "astar",
                         "heuristic": heuristic, "value": cost,
                         **asdict(stats), "median_ms": timed(call, repeats)})
            if name == "sample_weighted" and heuristic == "manhattan":
                cumulative, steps = 0, []
                for pos in path[1:]:
                    step_cost = enter_cost(grid, pos)
                    cumulative += step_cost
                    steps.append({"position": pos, "cost": step_cost, "cumulative": cumulative})
                data["problem2"] = {"path": path, "cost": cost, "steps": steps}

    for name, function in [("minimax", minimax), ("alphabeta", alpha_beta)]:
        for order in ("left", "reverse"):
            root = sample_tree()
            def call(function=function, order=order, root=root):
                stats = GameStats()
                return function(root, stats=stats, order=order), stats
            value, stats = call()
            rows.append({"problem": 3, "case": "sample_tree", "algorithm": name,
                         "order": order, "value": value, **asdict(stats),
                         "median_ms": timed(call, repeats)})

    matrices = {}
    for heuristic in ("zero", "manhattan"):
        distances, stats = distance_matrix(heuristic=heuristic)
        matrices[heuristic] = distances
        rows.append({"problem": 4, "case": "distance_matrix", "algorithm": "astar",
                     "heuristic": heuristic, "astar_calls": 30,
                     **asdict(stats), "median_ms": timed(
                         lambda h=heuristic: distance_matrix(heuristic=h), repeats)})
    assert matrices["zero"] == matrices["manhattan"]
    distances = matrices["manhattan"]
    data["problem4"] = {"labels": list(POINTS), "distances": distances,
                         "values": VALUES, "traces": {}, "first_target_values": {}}

    for depth in (2, 4):
        data["problem4"]["traces"][str(depth)] = optimal_trace(distances, depth)
        action_values = {}
        for target in VALUES:
            child = move(GameState(), target, distances)
            action_values[target] = solve_game(
                distances, depth - 1, "minimax", state=child)["best_value"]
        data["problem4"]["first_target_values"][str(depth)] = action_values
        for method in ("minimax", "alphabeta"):
            for order in ("natural", "reverse", "gain", "low_gain"):
                def call(depth=depth, method=method, order=order):
                    stats = GameStats()
                    return solve_game(distances, depth, method, order, stats=stats), stats
                result, stats = call()
                rows.append({"problem": 4, "case": "treasure_game", "algorithm": method,
                             "depth": depth, "order": order,
                             "best_target": result["best_target"],
                             "value": result["best_value"], **asdict(stats),
                             "median_ms": timed(call, repeats)})
        def cached_call(depth=depth):
            stats = GameStats()
            result = solve_game(distances, depth, "minimax", stats=stats, use_cache=True)
            return result, stats
        result, stats = cached_call()
        rows.append({"problem": 4, "case": "treasure_game", "algorithm": "minimax_cache",
                     "depth": depth, "order": "natural", "best_target": result["best_target"],
                     "value": result["best_value"], **asdict(stats),
                     "median_ms": timed(cached_call, repeats)})

    data["metrics"] = rows
    assert data["problem1"]["steps"] == 7
    assert data["problem2"]["cost"] == 7
    for row in rows:
        if row["problem"] == 3:
            assert row["value"] == 5
        if row["case"] == "treasure_game":
            assert (row["best_target"], row["value"]) == (
                ("T2", -1) if row["depth"] == 2 else ("T3", 3))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=51)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats 必须至少为 1")
    data = collect(args.repeats)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    (output / "experiment_results.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(dict.fromkeys(k for row in data["metrics"] for k in row))
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data["metrics"])
    lines = ["实验二实际运行结果", "题目一最短路径：" + str(data["problem1"]["path"]),
             "题目一步数：7", "题目二最小代价：7", "题目三最佳得分：5"]
    for depth in (2, 4):
        selected = next(r for r in data["metrics"] if r["case"] == "treasure_game"
                        and r["depth"] == depth and r["algorithm"] == "alphabeta"
                        and r["order"] == "natural")
        lines.append(f"题目四 H={depth}：{selected['best_target']}，收益差 {selected['value']}")
    lines.append(f"共保存 {len(data['metrics'])} 组指标；每组预热 3 次，计时 {args.repeats} 次取中位数。")
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
