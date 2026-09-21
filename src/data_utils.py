"""数据加载与划分。

课程提供的 train_data.csv 只含训练集，测试集为无标签文件；本文实验自行从
训练集中分层划出验证集（20%），验证集仅用于模型选择与超参数调整。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_SEED, TEST_CSV, TRAIN_CSV, VAL_SIZE


def load_data():
    """读取训练集与无标签测试集。

    Returns
    -------
    X_all : list[str]      训练文本
    y_all : np.ndarray     训练标签 (0-9)
    X_test : list[str]     待预测文本
    """
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    X_all = train_df["text"].astype(str).tolist()
    y_all = train_df["target"].to_numpy()
    X_test = test_df["text"].astype(str).tolist()
    return X_all, y_all, X_test


def split_train_val(X, y, val_size: float = VAL_SIZE, seed: int = RANDOM_SEED):
    """按类别分层划分训练集 / 验证集。

    分层可保证各子集类别比例与总体一致，降低小样本类别上的评估波动。
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=y
    )
    return X_tr, X_val, y_tr, y_val


def describe_dataset(y) -> pd.DataFrame:
    """返回类别分布表，用于报告中的数据集描述。"""
    counts = pd.Series(np.asarray(y)).value_counts().sort_index()
    return pd.DataFrame(
        {
            "类别": counts.index.to_numpy(),
            "样本数": counts.to_numpy(),
            "占比": (counts / counts.sum()).round(4).to_numpy(),
        }
    )
