"""全局配置：路径、随机种子与实验常量。

所有路径均相对项目根目录解析，避免依赖绝对路径，保证在干净环境下可复现。
"""

from pathlib import Path

# ---------------------------------------------------------------- 路径配置
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

TRAIN_CSV = DATA_DIR / "train_data.csv"
TEST_CSV = DATA_DIR / "test_data_unlabeled.csv"
PREDICTION_CSV = PROJECT_ROOT / "predictions.csv"

# ---------------------------------------------------------------- 实验常量
RANDOM_SEED = 42   # 全局随机种子，保证结果可复现
VAL_SIZE = 0.2     # 验证集比例（分层划分）
N_CLASSES = 10     # 新闻文本类别数

# 目录自动创建，便于脚本在任意位置运行
for _d in (DATA_DIR, RESULTS_DIR, FIGURE_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
