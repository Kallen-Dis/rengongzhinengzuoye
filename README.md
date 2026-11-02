# 实验二 搜索算法

代码仓库：[experiment2 分支](https://github.com/Kallen-Dis/rengongzhinengzuoye/tree/experiment2)。该分支仅包含实验二；`main` 分支保留实验一。

本项目实现四道题，并提供可重复执行的对照实验和实验报告。运行算法只需 Python 3.10 或更新版本，不需要安装第三方库。

## 快速运行

在本文件所在的“实验二”目录打开终端，执行：

```powershell
python run_experiments.py
python -m unittest discover -s code -p "test_*.py" -v
```

如果电脑上使用 `py` 启动 Python，把上面的 `python` 换成 `py -3`。
可用 `python run_experiments.py --repeats 101` 调整计时重复次数。

也可以分别运行：

```powershell
python code/problem1_maze.py
python code/problem2_weighted.py
python code/problem3_game_tree.py
python code/problem4_treasure.py
```

给定样例的结果：题目一步数 7；题目二最小代价 7；题目三最佳分数 5；题目四 H=2 选择 T2、收益差 -1，H=4 选择 T3、收益差 3。

## 文件说明

- `code/search_common.py`：A* 核心、路径还原和统计。
- `code/datasets.py`：题目给定地图、关键点位置、宝藏价值。
- `code/problem1_maze.py`：单位代价最短路径，公开函数 `astar_maze`。
- `code/problem2_weighted.py`：加权最短路，公开函数 `astar_weighted`；`astar_weighted_path` 同时返回路径和代价。
- `code/problem3_game_tree.py`：TreeNode、Minimax、Alpha-beta；默认从左到右。
- `code/problem4_treasure.py`：6×6 距离矩阵、不可变博弈状态、不同深度的决策。
- `code/test_algorithms.py`：独立 BFS、Dijkstra、行动排列枚举和固定种子随机验证。
- `run_experiments.py`：一键产生 42 组对照数据。
- `results/experiment_results.json`：完整数据、逐步收益、首步价值、运行环境。
- `results/metrics.csv`：方便筛选的性能数据。
- `results/summary.txt`：简短运行结果。
- `report/实验二报告.pdf`：5 页实验报告，满足不超过 5 页的要求。
- `report/实验二报告.md`：便于修改的报告正文。
- `build_report.py`：从 JSON 实际数据重新生成 PDF 和 Markdown。
- `学习说明.md`：适合初学者的代码阅读顺序和知识点。
- `original_code/`：修改前题目一、题目二的备份。

## 输入输出约定

坐标为 `(row, col)`，从 0 开始；只能上下左右移动。
题目一返回包含起点和终点的路径，无解返回 `[]`。
题目二返回进入各格的代价之和，出发不支付起点费用，进入 E 的费用是 0；无解返回 `-1`。
题目四的 A、B、T1 到 T4 的进入费用都是 1，不使用免费终点规则。
宝藏价值与地形代价分开存储；经过其他宝藏不自动收集，已取宝藏仍可通行，另一玩家不阻挡。
负收益合法，仍需按照 A、B、A、B 行动，不允许自行跳过回合。
额外地图中若当前玩家没有可达目标，程序约定以当前 U 终止；给定地图不触发此约定。

## 启发函数修正

题目一：`h = 曼哈顿距离`；`h=0` 是对照，相当于 Dijkstra（单位权时也能用 BFS）。
题目二：终点免费时，`h = max(曼哈顿距离 - 1, 0) * m`，m 为除目标外的最小通行代价。
若终点收费，则 `h = 曼哈顿距离 * m`，m 为所有通行格的最小代价。
如果其他格子含 0 代价，m=0，启发自动退化为 0，保持正确性。
原代码普通曼哈顿距离在“紧邻免费终点”时会高估，现已修正。

## 计数与计时口径

A* 的 `visited` 是有效出堆次数，包含终点；`expanded` 是枚举邻居次数，不含终点；`generated` 是入堆记录数，含起点及改进后的重复坐标；`stale` 为过期记录出堆数；`max_frontier` 为堆记录数峰值，不等于去重坐标数。

博弈的 `visited` 是递归进入次数，包含根和终止节点；`leaves` 是终止评价次数；`cutoffs` 仅统计确实跳过剩余分支的剪枝；`pruned_children` 统计跳过的直接子分支，不冒充所有被省略的后代数量。缓存命中也算一次递归进入。

计时使用 `perf_counter_ns`，每组预热 3 次，默认重复 51 次取中位数，以毫秒记录。统计结构、路径还原及排序开销包含在对应调用时间中；不含打印与文件写入。博弈计时不含距离矩阵的预计算，后者单独记录。解释最优行动序列的重复求解也不混入性能统计。

计时受机器和系统调度影响；节点数在相同代码、输入及访问顺序下可复现。A* 按 `(f, 坐标, 入堆时的g)` 打破平局，博弈相同最优值时保留当前顺序中的第一个动作。

## 拓展及限制

包含空旷地图、绕墙地图、固定种子随机地图、绕远但更便宜、终点免费、不可达和零代价地形。
博弈比较自然、逆序、即时高收益优先、即时低收益优先四种访问顺序；排序只影响访问顺序，不删除合法动作。
另比较完整 Minimax 的精确值缓存；本次四宝藏样例缓存命中为 0，不能据此宣称缓存一定加速。
Alpha-beta 的截断结果可能只是上下界，因此没有直接复用这种精确值缓存。

## 重新生成报告

已附 PDF，可直接阅读。仅在需要重建时安装 `reportlab`：

```powershell
python -m pip install reportlab
python run_experiments.py
python build_report.py
```

Windows 默认使用系统宋体与黑体；其他平台可用 `--font`、`--bold-font` 指定支持中文的 TTF 字体。PDF 与 Markdown 会读取同一份 JSON 数据同步生成；重新运行计时后，应同时重建报告。生成器检查页数，超过 5 页时会报错，提醒调整内容后重新生成。

报告页首附可点击的实验二分支链接；如迁移仓库，可用 `--repo-url` 显式修改。提交材料使用 `实验二提交材料.zip`，其中包含一份 5 页 PDF 报告、实验二代码和复现说明。
