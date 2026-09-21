# 当代人工智能 · 实验一：文本分类

新闻文本 10 分类实验：TF-IDF 特征工程 + 四种经典分类器的基线、调参与消融实验，最终以「原始文本 + TF-IDF(5000) + 逻辑回归(C=10)」在验证集达到 **0.9132** 准确率，5 折交叉验证 **0.9097 ± 0.0120**。

## 目录结构

```
project1-text-classification/
├── data/                  # 课程提供的数据（train_data.csv / test_data_unlabeled.csv）
├── src/                   # 模块化实现
│   ├── config.py          # 路径与全局常量（种子、验证集比例）
│   ├── data_utils.py      # 数据加载、分层划分
│   ├── preprocess.py      # 三种强度的文本清洗（none / light / heavy）
│   ├── features.py        # TF-IDF / 词袋 / 字符级拼接等向量化器与 Pipeline
│   ├── models.py          # NB / SVM / LR / MLP 工厂函数与搜索网格
│   ├── experiment.py      # 训练评估、结果落盘
│   └── plots.py           # 报告图表
├── run_experiments.py     # 实验主脚本（分阶段执行）
├── make_figures.py        # 图表生成入口
├── results/
│   ├── tables/            # 所有实验的原始结果表（CSV）
│   ├── figures/           # 报告用图（PNG）
│   └── best_config.json   # 最终方案配置与评估摘要
├── report/                # 实验报告（HTML + PDF）
└── predictions.csv        # 最终提交的测试集预测（2457 行，无表头）
```

## 一键复现

环境：Python ≥ 3.10（实测 3.13.2），安装依赖后按顺序执行：

```bash
pip install -r requirements.txt

python run_experiments.py --stage eda                  # 0. 数据概览
python run_experiments.py --stage baseline             # 1. 四模型基线
python run_experiments.py --stage tuning --group all   # 2. 超参数搜索（约 20 分钟）
python run_experiments.py --stage features --model svm --kernel linear --C 1.0   # 3. 特征消融（SVM）
python run_experiments.py --stage features --model lr --C 10.0                   # 3. 特征消融（LR）
python run_experiments.py --stage final --feature F1 --model lr --C 10.0         # 4. 最终方案 + 预测
python make_figures.py --figure all                    # 5. 生成全部图表
```

随机种子全局固定为 `42`，分层 8:2 划分验证集，测试集全程封存、仅最终预测一次。

## 结果摘要

| 阶段 | 最优配置 | 验证准确率 |
| --- | --- | --- |
| 基线（四模型，TF-IDF 5k + 去邮件头） | MLP(100,) | 0.8847 |
| 超参搜索 | LR C=10 | 0.8799 |
| 特征消融（LR） | F1 原始文本+TF-IDF(5k) | **0.9132** |
| 决赛圈 | NB+LR+SVM 软投票 | 0.9145（未采用，训练贵 60 倍） |
| **最终方案** | **原始文本 + TF-IDF(5000) + LR(C=10)** | 5 折 CV：**0.9097 ± 0.0120** |

主要发现：

1. 邮件头是信息不是噪声：保留原始文本（F1）比去头（F2）高 3.3 个百分点；
2. 词表大小是单项最大杠杆：5k → 3.3 万提升约 4 个百分点，超过换模型和调参的收益；
3. 过度清洗（去标点/数字/引文）明显有害（F3：0.7978）；
4. MLP 加宽加深均无提升，训练集拟合已达 0.999，瓶颈在泛化而非容量。

## 提交文件

- `predictions.csv`：2457 行、无表头、每行一个 0-9 整数，与课程样例格式一致
- 实验报告：`report/`（PDF 开头附本仓库链接）
