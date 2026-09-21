"""可视化模块：实验图表统一在这里生成，供实验报告直接引用。

所有图默认保存为 150 dpi 的 PNG 到 results/figures/。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面环境也能出图
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURE_DIR

# ---------------------------------------------------------------- 全局风格
def setup_style() -> None:
    """统一字体与风格；中文字体优先使用微软雅黑。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["axes.axisbelow"] = True


COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
MODEL_CN = {
    "nb": "朴素贝叶斯",
    "cnb": "补集朴素贝叶斯",
    "svm": "支持向量机",
    "lr": "逻辑回归",
    "mlp": "多层感知机",
}


def _save(fig, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"[保存图表] {path}")
    return path


# ---------------------------------------------------------------- 图 1：类别分布
def plot_class_distribution(dist: pd.DataFrame, name: str = "fig1_class_distribution.png"):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(dist["类别"].astype(str), dist["样本数"], color=COLORS[0], width=0.6)
    for b, v in zip(bars, dist["样本数"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 5, str(v), ha="center", fontsize=8)
    ax.set_xlabel("类别编号")
    ax.set_ylabel("样本数")
    ax.set_title("训练集类别分布（10 类，近似均衡）")
    ax.set_ylim(0, dist["样本数"].max() * 1.15)
    return _save(fig, name)


# ---------------------------------------------------------------- 图 2：基线对比
def plot_baseline(df: pd.DataFrame, name: str = "fig2_baseline.png"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    x = np.arange(len(df))
    w = 0.38
    axes[0].bar(x - w / 2, df["val_accuracy"], w, label="验证准确率", color=COLORS[0])
    axes[0].bar(x + w / 2, df["val_f1_macro"], w, label="宏平均 F1", color=COLORS[1])
    for i, (a, f) in enumerate(zip(df["val_accuracy"], df["val_f1_macro"])):
        axes[0].text(i - w / 2, a + 0.004, f"{a:.4f}", ha="center", fontsize=8)
        axes[0].text(i + w / 2, f + 0.004, f"{f:.4f}", ha="center", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([MODEL_CN.get(m, m) for m in df["model"]], fontsize=9)
    lo = min(df["val_accuracy"].min(), df["val_f1_macro"].min())
    axes[0].set_ylim(lo - 0.03, max(df["val_accuracy"].max(), df["val_f1_macro"].max()) + 0.03)
    axes[0].set_ylabel("指标值")
    axes[0].set_title("四模型基线表现（TF-IDF 5000 维）")
    axes[0].legend(fontsize=9)

    axes[1].barh([MODEL_CN.get(m, m) for m in df["model"]], df["fit_time_s"], color=COLORS[2])
    for i, t in enumerate(df["fit_time_s"]):
        axes[1].text(t, i, f" {t:.1f}s", va="center", fontsize=8)
    axes[1].set_xlabel("训练耗时 (s)")
    axes[1].set_title("训练代价对比")
    axes[1].set_xlim(0, df["fit_time_s"].max() * 1.3)
    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------- 图 3：调参曲线
def _param_series(df: pd.DataFrame, key: str) -> pd.Series:
    """从 params JSON 列中取出指定超参数的值。"""
    return df["params"].apply(lambda s: json.loads(s).get(key))


def plot_tuning_curves(tables: dict[str, pd.DataFrame], name: str = "fig3_tuning_curves.png"):
    """四联图：NB alpha、SVM C、LR C、MLP 结构。"""
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7))

    # (a) NB: alpha
    df = tables.get("nb")
    if df is not None:
        df = df.sort_values("val_accuracy", ascending=False)
        alphas = sorted(_param_series(df, "alpha").unique())
        accs = [_mean_for(df, "alpha", a, "val_accuracy") for a in alphas]
        f1s = [_mean_for(df, "alpha", a, "val_f1_macro") for a in alphas]
        axes[0, 0].plot(alphas, accs, "o-", label="准确率", color=COLORS[0])
        axes[0, 0].plot(alphas, f1s, "s--", label="宏平均 F1", color=COLORS[1])
        axes[0, 0].set_xscale("log")
        axes[0, 0].set_xlabel("alpha（拉普拉斯平滑强度）")
        axes[0, 0].set_title("(a) 朴素贝叶斯：alpha")

    # (b) SVM 线性核: C
    df = tables.get("svm")
    if df is not None:
        cs = sorted(_param_series(df, "C").unique())
        accs = [_mean_for(df, "C", c, "val_accuracy") for c in cs]
        f1s = [_mean_for(df, "C", c, "val_f1_macro") for c in cs]
        axes[0, 1].plot(cs, accs, "o-", label="准确率", color=COLORS[0])
        axes[0, 1].plot(cs, f1s, "s--", label="宏平均 F1", color=COLORS[1])
        axes[0, 1].set_xscale("log")
        axes[0, 1].set_xlabel("C（惩罚系数，线性核）")
        axes[0, 1].set_title("(b) 支持向量机：C")

    # (c) LR: C
    df = tables.get("lr")
    if df is not None:
        cs = sorted(_param_series(df, "C").unique())
        accs = [_mean_for(df, "C", c, "val_accuracy") for c in cs]
        f1s = [_mean_for(df, "C", c, "val_f1_macro") for c in cs]
        axes[1, 0].plot(cs, accs, "o-", label="准确率", color=COLORS[0])
        axes[1, 0].plot(cs, f1s, "s--", label="宏平均 F1", color=COLORS[1])
        axes[1, 0].set_xscale("log")
        axes[1, 0].set_xlabel("C（正则化强度倒数）")
        axes[1, 0].set_title("(c) 逻辑回归：C")

    # (d) MLP: 结构
    df = tables.get("mlp")
    if df is not None:
        labels = _param_series(df, "hidden_layer_sizes").astype(str) + " / α=" + _param_series(
            df, "alpha"
        ).astype(str)
        order = np.argsort(df["val_accuracy"].to_numpy())
        axes[1, 1].barh(
            np.array(labels)[order], df["val_accuracy"].to_numpy()[order], color=COLORS[4]
        )
        for i, (v, t) in enumerate(
            zip(df["val_accuracy"].to_numpy()[order], df["fit_time_s"].to_numpy()[order])
        ):
            axes[1, 1].text(v, i, f" {v:.4f} ({t:.0f}s)", va="center", fontsize=8)
        axes[1, 1].set_xlim(
            df["val_accuracy"].min() - 0.01, df["val_accuracy"].max() + 0.035
        )
        axes[1, 1].set_xlabel("验证准确率")
        axes[1, 1].set_title("(d) 多层感知机：结构与正则化")

    for ax in axes.flat:
        ax.legend(fontsize=8, loc="best") if ax.get_legend_handles_labels()[0] else None
    fig.suptitle("超参数调优曲线（验证集，其余条件固定）", fontsize=12)
    fig.tight_layout()
    return _save(fig, name)


def _mean_for(df: pd.DataFrame, key: str, value, col: str) -> float:
    mask = _param_series(df, key) == value
    return float(df.loc[mask, col].mean())


# ---------------------------------------------------------------- 图 4：特征消融
def plot_feature_ablation(df: pd.DataFrame, name: str = "fig4_feature_ablation.png"):
    df = df.sort_values("val_accuracy")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.barh(df["feature_name"], df["val_accuracy"], color=COLORS[0])
    base = df.loc[df["feature_name"].str.contains("基线"), "val_accuracy"]
    if len(base):
        ax.axvline(base.iloc[0], color=COLORS[3], ls="--", lw=1, label="基线特征配置")
    for b, v, t in zip(bars, df["val_accuracy"], df["fit_time_s"]):
        ax.text(v + 0.0005, b.get_y() + b.get_height() / 2, f"{v:.4f} ({t:.0f}s)", va="center", fontsize=8)
    ax.set_xlim(df["val_accuracy"].min() - 0.015, df["val_accuracy"].max() + 0.02)
    ax.set_xlabel("验证准确率")
    ax.set_title(f"特征工程消融对比（分类器固定：{df['model_label'].iloc[0]}）")
    ax.legend(fontsize=8)
    return _save(fig, name)


# ---------------------------------------------------------------- 图 5：混淆矩阵
def plot_confusion_matrix(cm, labels=None, name: str = "fig5_confusion_matrix.png"):
    cm = np.asarray(cm)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(cm.shape[0]))
    ax.set_yticks(range(cm.shape[0]))
    tick = labels if labels is not None else [str(i) for i in range(cm.shape[0])]
    ax.set_xticklabels(tick, fontsize=8)
    ax.set_yticklabels(tick, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(
                    j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.55 else "black",
                )
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("最佳模型验证集混淆矩阵（含计数，颜色为行归一化）")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label="行归一化比例")
    return _save(fig, name)


# ---------------------------------------------------------------- 图 6：每类指标
def plot_per_class(df: pd.DataFrame, name: str = "fig6_per_class_f1.png"):
    x = np.arange(len(df))
    w = 0.27
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.bar(x - w, df["precision"], w, label="精确率", color=COLORS[0])
    ax.bar(x, df["recall"], w, label="召回率", color=COLORS[1])
    ax.bar(x + w, df["f1"], w, label="F1", color=COLORS[2])
    ax.set_xticks(x)
    ax.set_xticklabels(df["class"].astype(str))
    ax.set_ylim(df[["precision", "recall", "f1"]].min().min() - 0.12, 1.02)
    ax.set_xlabel("类别编号")
    ax.set_ylabel("指标值")
    ax.set_title("最佳模型各类别 P / R / F1（验证集）")
    ax.legend(fontsize=9)
    return _save(fig, name)


# ---------------------------------------------------------------- 图 7：代价-效果
def plot_tradeoff(df: pd.DataFrame, name: str = "fig7_tradeoff.png"):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for i, row in enumerate(df.itertuples()):
        ax.scatter(row.fit_time_s, row.val_accuracy, s=70, color=COLORS[i % len(COLORS)])
        ax.annotate(
            f"{MODEL_CN.get(row.model, row.model)}\n({row.fit_time_s:.1f}s)",
            (row.fit_time_s, row.val_accuracy),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    ax.set_xlabel("训练耗时 (s)")
    ax.set_ylabel("验证准确率")
    ax.set_title("性能–训练代价权衡")
    ax.set_xlim(-5, df["fit_time_s"].max() * 1.25)
    ax.set_ylim(df["val_accuracy"].min() - 0.02, df["val_accuracy"].max() + 0.02)
    return _save(fig, name)


if __name__ == "__main__":
    setup_style()
    print("绘图模块就绪，可用 make_figures.py 调用。")
