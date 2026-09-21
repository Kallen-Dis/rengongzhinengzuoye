"""特征工程：把「文本清洗 + 向量化」封装为可整体 fit 的 sklearn 组件。

所有向量化器统计量（词表、IDF、文档频率）只在训练数据上 fit，验证集与测试集
仅调用 transform，从源头避免数据泄露。
"""

from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from .preprocess import clean_texts

# ---------------------------------------------------------------- 词表参数
# 英文停用词表（scikit-learn 内置），用于消融实验中的停用词对比
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORDS = sorted(ENGLISH_STOP_WORDS)


def build_vectorizer(kind: str = "tfidf", **params):
    """构造向量化器。

    kind 取值
    ---------
    tfidf      : 词级 TF-IDF（默认）
    count      : 词袋模型（词频计数），作为 TF-IDF 的对照
    tfidf_char : 字符级 TF-IDF（对拼写差异更鲁棒）
    tfidf_union: 词级 + 字符级 TF-IDF 的拼接（创新点之一）
    """
    if kind == "tfidf":
        return TfidfVectorizer(**params)
    if kind == "count":
        return CountVectorizer(**params)
    if kind == "tfidf_char":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **params)
    if kind == "tfidf_union":
        word = TfidfVectorizer(**params)
        # 字符级特征使用较少的维度即可提供互补信息
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=100_000,
            sublinear_tf=True,
        )
        return FeatureUnion([("word", word), ("char", char)])
    raise ValueError(f"未知向量化类型: {kind}")


def build_pipeline(feature_spec: dict, classifier):
    """把清洗强度、向量化器与分类器串成一条 Pipeline。

    Parameters
    ----------
    feature_spec : dict
        形如 {"preprocess": "light", "vectorizer": "tfidf", "params": {...}}
    classifier : sklearn 估计器
    """
    level = feature_spec.get("preprocess", "none")
    vectorizer = build_vectorizer(
        feature_spec.get("vectorizer", "tfidf"), **feature_spec.get("params", {})
    )
    return Pipeline(
        [
            (
                "clean",
                FunctionTransformer(
                    clean_texts, kw_args={"level": level}, validate=False
                ),
            ),
            ("features", vectorizer),
            ("clf", classifier),
        ]
    )


def feature_summary(feature_spec: dict) -> str:
    """生成特征配置的可读描述，用于结果表格与报告。"""
    return (
        f"{feature_spec.get('vectorizer', 'tfidf')}"
        f"[clean={feature_spec.get('preprocess', 'none')}, "
        f"{', '.join(f'{k}={v}' for k, v in feature_spec.get('params', {}).items())}]"
    )


# 实验中反复使用的基准特征配置（与 draft_main.py 提供的基线保持一致：max_features=5000）
BASE_FEATURE = {
    "preprocess": "light",
    "vectorizer": "tfidf",
    "params": {"max_features": 5000},
}
