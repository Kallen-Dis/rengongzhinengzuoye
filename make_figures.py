"""图表生成入口：从 results/ 中已保存的结果生成实验报告所需图表。

用法：
    python make_figures.py --figure all
    python make_figures.py --figure tuning
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.experiment import load_table
from src.plots import (
    plot_baseline,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_ablation,
    plot_per_class,
    plot_tradeoff,
    plot_tuning_curves,
    setup_style,
)


def _show(df: pd.DataFrame, cols: list[str], title: str, rename: dict | None = None) -> None:
    """在终端打印便于截图的结果表。"""
    view = df[cols].copy()
    if rename:
        view = view.rename(columns=rename)
    print(f"\n===== {title} =====")
    print(view.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def fig_eda() -> None:
    dist = pd.read_csv(RESULTS_DIR / "00_class_distribution.csv")
    plot_class_distribution(dist)


def fig_baseline() -> None:
    df = load_table("01_baseline.csv")
    _show(
        df,
        ["model_label", "val_accuracy", "val_f1_macro", "val_precision_macro", "val_recall_macro", "fit_time_s"],
        "四模型基线对比（TF-IDF 5000 维 + 去邮件头）",
        {
            "model_label": "模型",
            "val_accuracy": "验证准确率",
            "val_f1_macro": "宏平均F1",
            "val_precision_macro": "宏精确率",
            "val_recall_macro": "宏召回率",
            "fit_time_s": "训练耗时(s)",
        },
    )
    plot_baseline(df)
    plot_tradeoff(df)


def fig_tuning() -> None:
    tables = {}
    for g in ("nb", "svm", "lr", "mlp", "svm_rbf"):
        try:
            tables[g] = load_table(f"02_tuning_{g}.csv")
        except FileNotFoundError:
            print(f"[跳过] 未找到 02_tuning_{g}.csv")
    for g, df in tables.items():
        _show(
            df.sort_values("val_accuracy", ascending=False),
            ["config_desc", "val_accuracy", "val_f1_macro", "fit_time_s"],
            f"超参搜索：{df['group_title'].iloc[0]}",
            {
                "config_desc": "配置",
                "val_accuracy": "验证准确率",
                "val_f1_macro": "宏平均F1",
                "fit_time_s": "训练耗时(s)",
            },
        )
    plot_tuning_curves(tables)


def fig_ablation() -> None:
    files = sorted(RESULTS_DIR.glob("tables/03_feature_ablation*.csv"))
    for f in files:
        df = load_table(f.name)
        model = f.stem.replace("03_feature_ablation", "").strip("_") or "svm"
        _show(
            df.sort_values("val_accuracy", ascending=False),
            ["feature_name", "val_accuracy", "val_f1_macro", "fit_time_s"],
            f"特征工程消融（分类器：{df['model_label'].iloc[0]}）",
            {
                "feature_name": "特征配置",
                "val_accuracy": "验证准确率",
                "val_f1_macro": "宏平均F1",
                "fit_time_s": "训练耗时(s)",
            },
        )
        plot_feature_ablation(df, name=f"fig4_feature_ablation_{model}.png")


def fig_final() -> None:
    cm = np.load(RESULTS_DIR / "final_confusion_matrix.npy")
    per_class = pd.read_csv(RESULTS_DIR / "final_per_class.csv")
    cv = load_table("04_final_cv.csv")
    _show(cv, ["fold", "accuracy", "f1_macro", "fit_time_s"], "最终方案 5 折交叉验证",
          {"fold": "折", "accuracy": "准确率", "f1_macro": "宏平均F1", "fit_time_s": "训练耗时(s)"})
    _show(per_class, ["class", "precision", "recall", "f1"], "最佳模型各类别指标（验证集）",
          {"class": "类别", "precision": "精确率", "recall": "召回率", "f1": "F1"})
    plot_confusion_matrix(cm)
    plot_per_class(per_class)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成实验图表")
    parser.add_argument(
        "--figure",
        default="all",
        choices=["eda", "baseline", "tuning", "ablation", "final", "all"],
    )
    args = parser.parse_args()
    setup_style()

    if args.figure in ("eda", "all"):
        fig_eda()
    if args.figure in ("baseline", "all"):
        fig_baseline()
    if args.figure in ("tuning", "all"):
        fig_tuning()
    if args.figure in ("ablation", "all"):
        fig_ablation()
    if args.figure in ("final", "all"):
        fig_final()


if __name__ == "__main__":
    main()
