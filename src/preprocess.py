"""文本预处理：提供三种强度递增的清洗方案，用于消融实验对比。

清洗强度
--------
none  : 原始文本，不做任何处理（仅由 TfidfVectorizer 完成分词与小写化）。
light : 去掉邮件头（From/Subject/Newsgroups 等）与签名块，保留正文、引用与标点。
heavy : 在 light 基础上进一步去掉引用行、URL、邮箱、标点、数字与单字符碎片。

设计说明
--------
1. 清洗只作用于输入文本，不改变标签，不会引入标签噪声；
2. 所有清洗规则均为「逐样本」的无状态变换，因此在划分训练/验证集之后分别
   应用与在全部数据上应用等价，不存在跨集合的统计量泄露；
3. 与文本统计量相关的步骤（词表、IDF、min_df/max_df）全部交给
   TfidfVectorizer，且只在训练折上 fit，避免数据泄露。
"""

from __future__ import annotations

import re

# 邮件头字段（新闻组数据集为邮件/帖子格式，头部字段可能携带泄漏信息）
_HEADER_RE = re.compile(
    r"^(from|subject|to|cc|bcc|reply-to|sender|date|message-id|in-reply-to|references|"
    r"organization|lines|distribution|keywords|summary|newsgroups|followup-to|xref|path|"
    r"nntp-posting-host|nntp-posting-user|nntp-posting-date|news-software|x-newsreader|"
    r"x-mailer|x-complaints-to|content-type|content-length|originator|return-path|"
    r"received|expires|supersedes|control|approved|article-id)\s*:.*$",
    re.IGNORECASE | re.MULTILINE,
)
# 签名块：以 "-- " 单独成行开始，直到文末
_SIGNATURE_RE = re.compile(r"\n-{2,}\s*\n.*\Z", re.DOTALL)
# 引用行：以 > 或 | 开头
_QUOTE_RE = re.compile(r"^\s*[>|].*$", re.MULTILINE)
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\S+@\S+")
# 只保留英文字母与数字，其余一律替换为空格
_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]+")
_MULTISPACE_RE = re.compile(r"\s+")

PREPROCESS_LEVELS = ("none", "light", "heavy")


def clean_text(text: str, level: str = "none") -> str:
    """对单条文本执行指定强度的清洗。"""
    if level not in PREPROCESS_LEVELS:
        raise ValueError(f"未知的清洗强度: {level}，可选 {PREPROCESS_LEVELS}")
    if level == "none":
        return str(text)

    t = str(text)
    # ---- light 及 heavy 共有 ----
    t = _HEADER_RE.sub(" ", t)
    t = _SIGNATURE_RE.sub(" ", t)

    # ---- 仅 heavy ----
    if level == "heavy":
        t = _QUOTE_RE.sub(" ", t)
        t = _URL_RE.sub(" ", t)
        t = _EMAIL_RE.sub(" ", t)
        t = t.lower()
        t = _NON_ALNUM_RE.sub(" ", t)
        t = _MULTISPACE_RE.sub(" ", t).strip()
        # 丢弃长度小于 2 的碎片（纯噪声前缀等）
        t = " ".join(w for w in t.split() if len(w) > 1)
    return t


def clean_texts(texts, level: str = "none"):
    """对文本列表批量清洗，返回列表（供 sklearn Pipeline 使用）。"""
    if level == "none":
        return [str(t) for t in texts]
    return [clean_text(t, level) for t in texts]


if __name__ == "__main__":  # 手动检查清洗效果
    demo = "From: a@b.com\nSubject: Re: hello\nOrganization: test\n\nHello world!\n> quoted line\nSee http://x.y\n-- \nSig"
    for lv in PREPROCESS_LEVELS:
        print(f"[{lv}] {clean_text(demo, lv)!r}")
