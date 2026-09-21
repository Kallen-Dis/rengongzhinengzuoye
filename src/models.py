"""分类器工厂与超参数搜索空间定义。

四种课程要求的模型：
  nb  朴素贝叶斯 (MultinomialNB)
  svm 支持向量机 (SVC)
  lr  逻辑回归 (LogisticRegression)
  mlp 多层感知机 (MLPClassifier)
额外补充两个对照：
  cnb        ComplementNB（面向不平衡文本的贝叶斯变体，代价极低）
  ensemble  NB + LR + SVM 软投票集成（创新点）
"""

from __future__ import annotations

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .config import RANDOM_SEED


def make_classifier(name: str, **params):
    """按名称构造分类器；所有随机性来源均固定 random_state。"""
    name = name.lower()
    if name == "nb":
        return MultinomialNB(**{"alpha": 1.0, **params})
    if name == "cnb":
        return ComplementNB(**{"alpha": 1.0, **params})
    if name == "svm":
        # 文本数据高维稀疏，默认线性核；随机性仅来自概率估计，此处不启用
        return SVC(**{"kernel": "linear", "C": 1.0, "random_state": RANDOM_SEED, **params})
    if name == "lr":
        # sklearn 1.9 起 liblinear 不再支持多分类，这里使用 lbfgs（多项式损失）
        return LogisticRegression(
            **{
                "C": 1.0,
                "max_iter": 3000,
                "solver": "lbfgs",
                "random_state": RANDOM_SEED,
                **params,
            }
        )
    if name == "mlp":
        return MLPClassifier(
            **{
                "hidden_layer_sizes": (100,),
                "max_iter": 300,
                "random_state": RANDOM_SEED,
                **params,
            }
        )
    raise ValueError(f"未知模型: {name}")


def make_ensemble(estimators, voting: str = "soft"):
    """构造软投票集成。要求各基模型提供 predict_proba。"""
    return VotingClassifier(estimators=estimators, voting=voting, n_jobs=None)


# ---------------------------------------------------------------- 搜索空间
# 每个条目为 (说明, 模型名, 参数字典)；键名以 clf__ 前缀对应 Pipeline 中的分类器
SVM_LINEAR_GRID = [
    (f"C={c}", "svm", {"kernel": "linear", "C": c}) for c in (0.1, 1.0, 10.0, 100.0)
]
SVM_RBF_GRID = [
    (f"rbf C={c} gamma={g}", "svm", {"kernel": "rbf", "C": c, "gamma": g})
    for c in (1.0, 10.0)
    for g in ("scale", 0.01)
]
NB_GRID = [(f"alpha={a}", "nb", {"alpha": a}) for a in (0.01, 0.1, 0.5, 1.0)]
LR_GRID = [(f"C={c}", "lr", {"C": c}) for c in (0.1, 1.0, 10.0, 100.0)]
MLP_GRID = [
    (desc, "mlp", params)
    for desc, params in [
        ("hidden=(100,), alpha=1e-4", {"hidden_layer_sizes": (100,), "alpha": 1e-4}),
        ("hidden=(200,), alpha=1e-4", {"hidden_layer_sizes": (200,), "alpha": 1e-4}),
        ("hidden=(100,50), alpha=1e-4", {"hidden_layer_sizes": (100, 50), "alpha": 1e-4}),
        ("hidden=(400,), alpha=1e-4", {"hidden_layer_sizes": (400,), "alpha": 1e-4}),
        ("hidden=(200,), alpha=1e-2", {"hidden_layer_sizes": (200,), "alpha": 1e-2}),
    ]
]

MODEL_LABELS = {
    "nb": "朴素贝叶斯 (MultinomialNB)",
    "cnb": "补集朴素贝叶斯 (ComplementNB)",
    "svm": "支持向量机 (SVC)",
    "lr": "逻辑回归 (LogisticRegression)",
    "mlp": "多层感知机 (MLPClassifier)",
    "ensemble": "软投票集成 (NB+LR+SVM)",
}
