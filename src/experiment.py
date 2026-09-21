"""实验执行与评估工具：训练、验证、指标计算与结果落盘。

统一的评估函数保证「一次实验只改变需要研究的因素」：
只要传入的 Pipeline 相同、数据划分相同，比较结果即具有可比性。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import RANDOM_SEED, TABLE_DIR


def reset_seed(seed: int = RANDOM_SEED) -> None:
    """在每次实验前重置全局随机种子，降低运行顺序带来的干扰。"""
    import random

    random.seed(seed)
    np.random.seed(seed)


def evaluate_pipeline(pipeline, X_tr, y_tr, X_val, y_val, collect_detail: bool = False) -> dict:
    """在训练集上拟合、在验证集上评估，返回指标字典。"""
    t0 = time.perf_counter()
    pipeline.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = pipeline.predict(X_val)
    predict_time = time.perf_counter() - t1

    result: dict[str, Any] = {
        "val_accuracy": accuracy_score(y_val, y_pred),
        "val_f1_macro": f1_score(y_val, y_pred, average="macro"),
        "val_f1_weighted": f1_score(y_val, y_pred, average="weighted"),
        "val_precision_macro": precision_score(y_val, y_pred, average="macro"),
        "val_recall_macro": recall_score(y_val, y_pred, average="macro"),
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
    }
    if collect_detail:
        result["confusion_matrix"] = confusion_matrix(y_val, y_pred)
        result["per_class_f1"] = f1_score(y_val, y_pred, average=None)
        result["per_class_recall"] = recall_score(y_val, y_pred, average=None)
        result["per_class_precision"] = precision_score(y_val, y_pred, average=None)
        result["y_pred"] = y_pred
    return result


def save_table(df: pd.DataFrame, filename: str) -> None:
    """把结果表保存到 results/tables/，统一保留 4 位小数。"""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / filename
    df.to_csv(path, index=False, float_format="%.4f", encoding="utf-8-sig")
    print(f"[保存表格] {path}")


def load_table(filename: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / filename)


def format_table(df: pd.DataFrame, columns: list[str], rename: dict | None = None) -> str:
    """把结果表格式化为 Markdown，方便直接粘贴进实验报告。"""
    view = df[columns].copy()
    if rename:
        view = view.rename(columns=rename)
    return view.to_markdown(index=False, floatfmt=".4f")
