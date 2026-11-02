"""独立基准验证：BFS、Dijkstra、行动排列枚举；固定种子随机测试。"""
from collections import deque
import heapq
import itertools
import random
import unittest

from datasets import MAZE, WEIGHTED, TREASURE_GRID, POINTS, VALUES
from problem1_maze import astar_maze
from problem2_weighted import astar_weighted, astar_weighted_path, make_heuristic
from problem3_game_tree import (TreeNode, sample_tree, tree_from_dict,
                                minimax, alpha_beta)
from problem4_treasure import GameState, distance_matrix, solve_game, move


def independent_dijkstra(grid, start, goal):
    """独立实现，不调用 A*、其启发函数或代价解析函数。"""
    def cost(pos):
        cell = grid[pos[0]][pos[1]]
        if cell == 'X':
            return None
        if cell == 'E':
            return 0
        if cell in ('S', 'A', 'B', 'T1', 'T2', 'T3', 'T4'):
            return 1
        return int(cell)
    if cost(start) is None or cost(goal) is None:
        return -1
    queue, best = [(0, start)], {start: 0}
    while queue:
        total, pos = heapq.heappop(queue)
        if total != best[pos]:
            continue
        if pos == goal:
            return total
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nxt = pos[0] + dr, pos[1] + dc
            if not (0 <= nxt[0] < len(grid) and 0 <= nxt[1] < len(grid[0])):
                continue
            step = cost(nxt)
            if step is not None and total + step < best.get(nxt, float('inf')):
                best[nxt] = total + step
                heapq.heappush(queue, (total + step, nxt))
    return -1


def independent_game(distances, depth):
    """枚举全部不重复宝藏序列，再按前缀从底层归约；不使用被测 move。"""
    layers = {}
    treasures = tuple(VALUES)
    for sequence in itertools.permutations(treasures, depth):
        positions, scores = ['A', 'B'], [0, 0]
        for i, target in enumerate(sequence):
            player = i % 2
            scores[player] += VALUES[target] - distances[positions[player]][target]
            positions[player] = target
        layers[sequence] = scores[0] - scores[1]
    for level in range(depth - 1, -1, -1):
        reduced = {}
        for prefix in itertools.permutations(treasures, level):
            choices = [layers[prefix + (t,)] for t in treasures if t not in prefix]
            reduced[prefix] = max(choices) if level % 2 == 0 else min(choices)
        layers = reduced
    return layers[()]


class SearchTests(unittest.TestCase):
    def check_path(self, grid, path, start, goal):
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        for r, c in path:
            self.assertTrue(0 <= r < len(grid) and 0 <= c < len(grid[0]))
            self.assertNotEqual(grid[r][c], 'X')
        for a, b in zip(path, path[1:]):
            self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)

    def test_maze_sample_and_bfs(self):
        path = astar_maze(MAZE, (0, 0), (3, 4))
        self.check_path(MAZE, path, (0, 0), (3, 4))
        queue, seen = deque([((0, 0), 0)]), {(0, 0)}
        while queue:
            (r, c), steps = queue.popleft()
            if (r, c) == (3, 4):
                self.assertEqual(len(path) - 1, steps)
                break
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = r + dr, c + dc
                if (0 <= p[0] < 5 and 0 <= p[1] < 5 and p not in seen
                        and MAZE[p[0]][p[1]] != 'X'):
                    seen.add(p)
                    queue.append((p, steps + 1))
        self.assertEqual(len(path) - 1, 7)

    def test_weighted_sample(self):
        path, cost = astar_weighted_path(WEIGHTED, (0, 0), (4, 4))
        self.check_path(WEIGHTED, path, (0, 0), (4, 4))
        self.assertEqual(cost, independent_dijkstra(WEIGHTED, (0, 0), (4, 4)))
        self.assertEqual(cost, 7)

    def test_boundaries_and_no_path(self):
        self.assertEqual(astar_maze([['S', 'X', 'E']], (0, 0), (0, 2)), [])
        self.assertEqual(astar_weighted([['S', 'X', 'E']], (0, 0), (0, 2)), -1)
        self.assertEqual(astar_maze([['S']], (0, 0), (0, 0)), [(0, 0)])
        self.assertEqual(astar_weighted([['S']], (0, 0), (0, 0)), 0)
        self.assertEqual(astar_maze([['X']], (0, 0), (0, 0)), [])
        with self.assertRaises(ValueError):
            astar_maze([], (0, 0), (0, 0))
        with self.assertRaises(ValueError):
            astar_weighted([['S'], ['1', 'E']], (0, 0), (1, 1))

    def test_free_goal_and_detour(self):
        self.assertEqual(astar_weighted([['S', 'E']], (0, 0), (0, 1)), 0)
        self.assertEqual(make_heuristic([['S', 'E']], (0, 1))((0, 0)), 0)
        grid = [['S', '9', 'E'], ['1', '1', '1']]
        path, cost = astar_weighted_path(grid, (0, 0), (0, 2))
        self.assertEqual((len(path) - 1, cost), (4, 3))
        self.assertEqual(astar_weighted([['S', '0', '0', 'E']], (0, 0), (0, 3)), 0)

    def test_random_weighted_and_heuristic(self):
        rng = random.Random(1431)
        for index in range(60):
            # 一半地图全为正代价，确保也测试非零启发；另一半允许零代价。
            lower = 1 if index % 2 == 0 else 0
            grid = [[('X' if rng.random() < 0.2 else str(rng.randint(lower, 9)))
                     for _ in range(6)] for _ in range(6)]
            grid[0][0], grid[5][5] = 'S', 'E'
            reference = independent_dijkstra(grid, (0, 0), (5, 5))
            with self.subTest(index=index):
                for mode in ('zero', 'manhattan'):
                    path, cost = astar_weighted_path(grid, (0, 0), (5, 5), mode)
                    self.assertEqual(cost, reference)
                    if path:
                        self.check_path(grid, path, (0, 0), (5, 5))
                h = make_heuristic(grid, (5, 5))
                for r in range(6):
                    for c in range(6):
                        if grid[r][c] != 'X':
                            exact = independent_dijkstra(grid, (r, c), (5, 5))
                            if exact >= 0:
                                self.assertLessEqual(h((r, c)), exact)
                            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                                nr, nc = r + dr, c + dc
                                if (0 <= nr < 6 and 0 <= nc < 6
                                        and grid[nr][nc] != 'X'):
                                    cell = grid[nr][nc]
                                    step = 0 if cell == 'E' else (1 if cell == 'S' else int(cell))
                                    self.assertLessEqual(h((r, c)), step + h((nr, nc)))


class GameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.distances, _ = distance_matrix()

    def test_tree_sample_orders_and_json(self):
        for order in ('left', 'reverse'):
            self.assertEqual(minimax(sample_tree(), order=order), 5)
            self.assertEqual(alpha_beta(sample_tree(), order=order), 5)
        node = tree_from_dict({'tree_structure': {'children': [{'value': 0}, {'value': -3}]}})
        self.assertEqual(alpha_beta(node), 0)

    def test_random_trees(self):
        rng = random.Random(2026)
        def create(depth):
            if depth == 0 or rng.random() < 0.15:
                return TreeNode(value=rng.randint(-20, 20))
            return TreeNode(children=[create(depth - 1) for _ in range(rng.randint(1, 4))])
        for _ in range(50):
            root = create(4)
            exact = minimax(root)
            for order in ('left', 'reverse'):
                self.assertEqual(alpha_beta(root, order=order), exact)

    def test_all_distances_independently(self):
        for source, start in POINTS.items():
            for target, goal in POINTS.items():
                self.assertEqual(self.distances[source][target],
                                 independent_dijkstra(TREASURE_GRID, start, goal))

    def test_depth_results_against_enumeration(self):
        for depth, target, value in ((2, 'T2', -1), (4, 'T3', 3)):
            self.assertEqual(independent_game(self.distances, depth), value)
            for method in ('minimax', 'alphabeta'):
                for order in ('natural', 'reverse', 'gain', 'low_gain'):
                    self.assertEqual(solve_game(self.distances, depth, method, order),
                                     {'best_target': target, 'best_value': value})
            self.assertEqual(solve_game(self.distances, depth, 'minimax', use_cache=True),
                             {'best_target': target, 'best_value': value})

    def test_state_and_negative_gain(self):
        original = GameState()
        after = move(original, 'T1', self.distances)
        self.assertEqual(after.score_a, -1)
        self.assertEqual(after.pos_a, 'T1')
        self.assertEqual(after.pos_b, 'B')
        self.assertEqual(after.turn, 'B')
        self.assertNotIn('T1', after.remaining)
        self.assertEqual(len(after.remaining), 3)
        self.assertEqual(len(original.remaining), 4)
        self.assertEqual(solve_game(self.distances, 0), {'best_target': None, 'best_value': 0})
        empty = GameState(remaining=(), score_a=2, score_b=-3)
        self.assertEqual(solve_game(self.distances, state=empty)['best_value'], 5)
        with self.assertRaises(ValueError):
            move(after, 'T1', self.distances)


if __name__ == '__main__':
    unittest.main(verbosity=2)
