"""实验一主脚本：按阶段执行完整实验流程。

用法示例
--------
python run_experiments.py --stage baseline          # 阶段1：四模型基线对比
python run_experiments.py --stage tuning --group nb # 阶段2：单个模型的超参搜索
python run_experiments.py --stage tuning --group all
python run_experiments.py --stage features          # 阶段3：特征工程消融
python run_experiments.py --stage final             # 阶段4：最终模型 5 折CV + 预测测试集
python run_experiments.py --stage all               # 一键复现全部实验

设计原则：所有阶段共用同一份数据划分（分层、seed=42），保证不同实验之间可比。
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.config import PREDICTION_CSV, RANDOM_SEED, RESULTS_DIR, VAL_SIZE
from src.data_utils import describe_dataset, load_data, split_train_val
from src.experiment import evaluate_pipeline, reset_seed, save_table
from src.features import BASE_FEATURE, build_pipeline, feature_summary
from src.models import (
    LR_GRID,
    MLP_GRID,
    MODEL_LABELS,
    NB_GRID,
    SVM_LINEAR_GRID,
    SVM_RBF_GRID,
    make_classifier,
)

BEST_CONFIG_PATH = RESULTS_DIR / "best_config.json"


# ---------------------------------------------------------------- 公共工具
def load_split():
    """加载数据并分层划分，返回 (X_tr, X_val, y_tr, y_val, X_test, y_all)。"""
    X_all, y_all, X_test = load_data()
    X_tr, X_val, y_tr, y_val = split_train_val(X_all, y_all, VAL_SIZE, RANDOM_SEED)
    print(
        f"[数据] 训练集 {len(X_tr)} 条 | 验证集 {len(X_val)} 条 | "
        f"待预测测试集 {len(X_test)} 条 | 类别数 {len(np.unique(y_all))}"
    )
    return X_tr, X_val, y_tr, y_val, X_test, X_all, y_all


def run_one(X_tr, y_tr, X_val, y_val, feature_spec: dict, model_name: str, params: dict) -> dict:
    """执行一次「特征配置 + 模型 + 超参」的实验，返回验证集指标。"""
    reset_seed(RANDOM_SEED)
    clf = make_classifier(model_name, **params)
    pipe = build_pipeline(feature_spec, clf)
    res = evaluate_pipeline(pipe, X_tr, y_tr, X_val, y_val)
    res.update(
        {
            "model": model_name,
            "model_label": MODEL_LABELS.get(model_name, model_name),
            "params": json.dumps(params, ensure_ascii=False),
            "feature": feature_summary(feature_spec),
        }
    )
    return res


# ---------------------------------------------------------------- 阶段 1：基线
def stage_baseline():
    """四个模型在「默认超参 + 统一特征配置」下的基线表现。

    基线目的：跑通完整流程，得到后续改进的比较起点。
    """
    X_tr, X_val, y_tr, y_val, *_ = load_split()
    rows = []
    for name in ("nb", "svm", "lr", "mlp"):
        res = run_one(X_tr, y_tr, X_val, y_val, BASE_FEATURE, name, {})
        rows.append(res)
        print(
            f"[基线] {MODEL_LABELS[name]:<28} acc={res['val_accuracy']:.4f} "
            f"macroF1={res['val_f1_macro']:.4f} fit={res['fit_time_s']:.1f}s"
        )
    df = pd.DataFrame(rows)
    save_table(df, "01_baseline.csv")
    return df


# ---------------------------------------------------------------- 阶段 2：调参
def stage_tuning(group: str = "all"):
    """逐个模型进行单变量超参数搜索（其余条件固定为基线特征配置）。"""
    X_tr, X_val, y_tr, y_val, *_ = load_split()
    groups = {
        "nb": ("朴素贝叶斯 alpha", NB_GRID),
        "svm": ("SVM 线性核 C", SVM_LINEAR_GRID),
        "lr": ("逻辑回归 C", LR_GRID),
        "mlp": ("MLP 结构/正则化", MLP_GRID),
        "svm_rbf": ("SVM RBF 核", SVM_RBF_GRID),
    }
    todo = list(groups) if group == "all" else [group]
    all_rows = []
    for g in todo:
        title, grid = groups[g]
        rows = []
        print(f"\n=== 超参搜索：{title} ===")
        for desc, model_name, params in grid:
            res = run_one(X_tr, y_tr, X_val, y_val, BASE_FEATURE, model_name, params)
            res.update({"group": g, "group_title": title, "config_desc": desc})
            rows.append(res)
            print(
                f"  {desc:<28} acc={res['val_accuracy']:.4f} "
                f"macroF1={res['val_f1_macro']:.4f} fit={res['fit_time_s']:.1f}s"
            )
        df = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False)
        save_table(df, f"02_tuning_{g}.csv")
        all_rows.append(df)
        best = df.iloc[0]
        print(f"  -> 最佳: {best['config_desc']} (acc={best['val_accuracy']:.4f})")
    return pd.concat(all_rows, ignore_index=True) if all_rows else None


# ---------------------------------------------------------------- 阶段 3：特征消融
FEATURE_SPECS = {
    "F1 原始文本+TFIDF(5k)": {
        "preprocess": "none",
        "vectorizer": "tfidf",
        "params": {"max_features": 5000},
    },
    "F2 去邮件头+TFIDF(5k)【基线】": dict(BASE_FEATURE),
    "F3 强度清洗+TFIDF(5k)": {
        "preprocess": "heavy",
        "vectorizer": "tfidf",
        "params": {"max_features": 5000},
    },
    "F4 词袋计数(5k)": {
        "preprocess": "light",
        "vectorizer": "count",
        "params": {"max_features": 5000},
    },
    "F5 去掉停用词+TFIDF(5k)": {
        "preprocess": "light",
        "vectorizer": "tfidf",
        "params": {"max_features": 5000, "stop_words": "english"},
    },
    "F6 二元词组+TFIDF(5k)": {
        "preprocess": "light",
        "vectorizer": "tfidf",
        "params": {"max_features": 5000, "ngram_range": (1, 2)},
    },
    "F7 sublinear TF+min_df=2(20k)": {
        "preprocess": "light",
        "vectorizer": "tfidf",
        "params": {"max_features": 20000, "min_df": 2, "sublinear_tf": True},
    },
    "F8 全词表 sublinear TF": {
        "preprocess": "light",
        "vectorizer": "tfidf",
        "params": {"min_df": 2, "sublinear_tf": True},
    },
    "F9 词+字符级TFIDF拼接": {
        "preprocess": "light",
        "vectorizer": "tfidf_union",
        "params": {"max_features": 50000, "min_df": 2, "sublinear_tf": True},
    },
    "F10 强度清洗+词+字符拼接": {
        "preprocess": "heavy",
        "vectorizer": "tfidf_union",
        "params": {"max_features": 50000, "min_df": 2, "sublinear_tf": True},
    },
}


def stage_features(model_name: str = "svm", params: dict | None = None):
    """特征工程消融：固定分类器，只改变文本清洗与向量化方式。"""
    X_tr, X_val, y_tr, y_val, *_ = load_split()
    params = params or {"kernel": "linear", "C": 1.0}
    rows = []
    print(f"\n=== 特征消融（分类器固定为 {MODEL_LABELS[model_name]} {params}）===")
    for name, spec in FEATURE_SPECS.items():
        res = run_one(X_tr, y_tr, X_val, y_val, spec, model_name, params)
        res.update({"feature_name": name, "dim_note": feature_summary(spec)})
        rows.append(res)
        print(
            f"  {name:<24} acc={res['val_accuracy']:.4f} "
            f"macroF1={res['val_f1_macro']:.4f} fit={res['fit_time_s']:.1f}s"
        )
    df = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False)
    save_table(df, f"03_feature_ablation_{model_name}.csv")
    return df


def resolve_feature(key: str) -> dict:
    """按短名（F1~F10）或全名解析特征配置。"""
    if key in FEATURE_SPECS:
        return FEATURE_SPECS[key]
    matched = [k for k in FEATURE_SPECS if k.startswith(key + " ")]
    if len(matched) != 1:
        raise KeyError(f"未知特征配置: {key}，可选 {list(FEATURE_SPECS)}")
    return FEATURE_SPECS[matched[0]]


# ---------------------------------------------------------------- 阶段 4：最终模型
def stage_final(feature_key: str = "F8", model_name: str = "svm", params: dict | None = None):
    """最终方案：5 折交叉验证估计泛化能力 → 全训练集重训 → 预测测试集。

    说明：测试集为无标签文件，始终不参与任何选择过程；此处仅做一次前向预测。
    """
    from sklearn.model_selection import StratifiedKFold, cross_validate

    X_tr, X_val, y_tr, y_val, X_test, X_all, y_all = load_split()
    spec = resolve_feature(feature_key)
    params = params or {"kernel": "linear", "C": 1.0}
    print(f"\n=== 最终方案：{feature_key} + {MODEL_LABELS[model_name]} {params} ===")

    # (1) 5 折交叉验证：在「全训练集」上估计泛化能力与波动
    reset_seed(RANDOM_SEED)
    pipe = build_pipeline(spec, make_classifier(model_name, **params))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_res = cross_validate(
        pipe, X_all, y_all, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=1
    )
    cv_df = pd.DataFrame(
        {
            "fold": np.arange(1, 6),
            "accuracy": cv_res["test_accuracy"],
            "f1_macro": cv_res["test_f1_macro"],
            "fit_time_s": cv_res["fit_time"],
        }
    )
    save_table(cv_df, "04_final_cv.csv")
    print(
        f"[5折CV] accuracy = {cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f} | "
        f"macroF1 = {cv_df['f1_macro'].mean():.4f} ± {cv_df['f1_macro'].std():.4f}"
    )

    # (2) 在全部训练数据上重训，并记录训练-验证差异（用于过拟合判断）
    val_res = run_one(X_tr, y_tr, X_val, y_val, spec, model_name, params)
    print(
        f"[留出验证] acc={val_res['val_accuracy']:.4f} macroF1={val_res['val_f1_macro']:.4f}"
    )

    # (3) 用全部训练数据拟合最终模型并预测测试集
    reset_seed(RANDOM_SEED)
    final_pipe = build_pipeline(spec, make_classifier(model_name, **params))
    final_pipe.fit(X_all, y_all)

    # (4) 记录最终方案的配置与训练集内表现（不做任何基于测试集的调整）
    train_pred = final_pipe.predict(X_all)
    train_acc = float((train_pred == y_all).mean())
    predictions = final_pipe.predict(X_test)
    pd.DataFrame(predictions).to_csv(PREDICTION_CSV, index=False, header=False)
    print(f"[预测] 已保存 {PREDICTION_CSV} （{len(predictions)} 条）")
    print(f"[拟合诊断] 全训练集拟合准确率 = {train_acc:.4f}")

    summary = {
        "feature_key": feature_key,
        "feature_spec": spec,
        "model": model_name,
        "params": params,
        "cv_accuracy_mean": float(cv_df["accuracy"].mean()),
        "cv_accuracy_std": float(cv_df["accuracy"].std()),
        "cv_f1_macro_mean": float(cv_df["f1_macro"].mean()),
        "cv_f1_macro_std": float(cv_df["f1_macro"].std()),
        "holdout_val_accuracy": float(val_res["val_accuracy"]),
        "holdout_val_f1_macro": float(val_res["val_f1_macro"]),
        "train_accuracy": train_acc,
        "prediction_distribution": {
            str(k): int(v) for k, v in pd.Series(predictions).value_counts().sort_index().items()
        },
        "n_train": int(len(X_all)),
        "n_test": int(len(X_test)),
    }
    BEST_CONFIG_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[保存配置] {BEST_CONFIG_PATH}")

    # (5) 保存混淆矩阵与各类别指标（报告用）
    detail = evaluate_pipeline(
        build_pipeline(spec, make_classifier(model_name, **params)),
        X_tr,
        y_tr,
        X_val,
        y_val,
        collect_detail=True,
    )
    np.save(RESULTS_DIR / "final_confusion_matrix.npy", detail["confusion_matrix"])
    pd.DataFrame(
        {
            "class": np.arange(len(detail["per_class_f1"])),
            "f1": detail["per_class_f1"],
            "precision": detail["per_class_precision"],
            "recall": detail["per_class_recall"],
        }
    ).to_csv(RESULTS_DIR / "final_per_class.csv", index=False, float_format="%.4f")
    np.save(RESULTS_DIR / "final_y_pred.npy", detail["y_pred"])
    np.save(RESULTS_DIR / "final_y_val.npy", y_val)

    # (6) 训练集/验证集规模与类别分布，供报告使用
    pd.DataFrame({"split": ["train", "val"], "n": [len(X_tr), len(X_val)]}).to_csv(
        RESULTS_DIR / "final_split_size.csv", index=False
    )
    return summary


# ---------------------------------------------------------------- 阶段 0：数据概览
def stage_eda():
    """导出数据集描述性统计，用于报告「数据集介绍」一节。"""
    X_all, y_all, X_test = load_data()
    dist = describe_dataset(y_all)
    dist.to_csv(RESULTS_DIR / "00_class_distribution.csv", index=False, float_format="%.4f")
    text_len = pd.Series([len(t) for t in X_all])
    stats = pd.DataFrame(
        {
            "指标": ["样本数", "字符数均值", "字符数中位数", "字符数标准差", "最短", "最长"],
            "训练集": [
                len(X_all),
                text_len.mean(),
                text_len.median(),
                text_len.std(),
                text_len.min(),
                text_len.max(),
            ],
        }
    )
    stats.to_csv(RESULTS_DIR / "00_text_length.csv", index=False, float_format="%.1f")
    pd.DataFrame({"n_test": [len(X_test)]}).to_csv(RESULTS_DIR / "00_test_size.csv", index=False)
    print(dist.to_string(index=False))
    print(stats.to_string(index=False))
    return dist


# ---------------------------------------------------------------- 入口
def main():
    parser = argparse.ArgumentParser(description="当代人工智能实验一：文本分类")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["eda", "baseline", "tuning", "features", "final", "all"],
    )
    parser.add_argument("--group", default="all", help="tuning 阶段的模型组")
    parser.add_argument("--feature", default="F8", help="final 阶段使用的特征键（F1~F10）或名称")
    parser.add_argument("--model", default="svm", help="final/features 阶段使用的模型")
    parser.add_argument("--kernel", default="linear")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=None)
    args = parser.parse_args()

    t_start = time.perf_counter()
    if args.stage in ("eda", "all"):
        print("=" * 70, "\n阶段 0：数据集概览\n" + "=" * 70)
        stage_eda()
    if args.stage in ("baseline", "all"):
        print("=" * 70, "\n阶段 1：四模型基线对比\n" + "=" * 70)
        stage_baseline()
    if args.stage in ("tuning", "all"):
        print("=" * 70, "\n阶段 2：超参数搜索\n" + "=" * 70)
        stage_tuning(args.group)
    def build_params():
        if args.model == "lr":
            return {"C": args.C}
        if args.model in ("nb", "cnb"):
            return {"alpha": args.alpha} if args.alpha is not None else {}
        return {"kernel": args.kernel, "C": args.C}

    if args.stage in ("features", "all"):
        print("=" * 70, "\n阶段 3：特征工程消融\n" + "=" * 70)
        stage_features(args.model, build_params())
    if args.stage in ("final", "all"):
        print("=" * 70, "\n阶段 4：最终方案与测试集预测\n" + "=" * 70)
        stage_final(args.feature, args.model, build_params())
    print(f"\n全部完成，总耗时 {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
