# 实验三 CIFAR-10 图像分类

从零训练基础CNN和改进CNN，使用独立验证集选取checkpoint，最后评估完整测试集。

## 本次实测结果

2026年11月6日在RTX 4070 Laptop GPU上完成两个模型各30轮训练：

| 模型 | 最佳轮次 | 验证准确率 | 测试准确率 |
| --- | --- | --- | --- |
| 基础CNN | 28 | 85.00% | 84.86% |
| 残差与SE改进CNN | 27 | 88.80% | 87.77% |

按验证集选出的改进模型在测试集上提高2.91个百分点。8组结果一致性检查全部通过；重新加载两个检查点后，完整测试集准确率均与原记录一致。报告共5页。提交包包含代码、权重、结果和报告，不包含原始数据集。

## 运行

课程提交要求：在课程群提供的第三次作业链接上传 `report/实验三报告.pdf`，报告不超过5页，开头已附实验三GitHub分支链接；代码及README保存在GitHub。ZIP仅用于完整材料备份，不作为本次课程平台的上传格式。

推荐Python 3.13及NVIDIA GPU。安装GPU版本时遵循PyTorch官方安装说明：

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy matplotlib pillow reportlab
python train.py --epochs 30 --batch 256
python make_figures.py
python verify_results.py
python build_report.py
```

当前课程环境可用上一级 `.venv/Scripts/python.exe` 运行前三个脚本。PDF生成依赖reportlab和Windows宋体/黑体，其他平台需调整build_report.py中的字体路径。

## 数据与规范

- CIFAR-10官方训练集50000张，按类固定划分45000训练、5000验证；测试10000张保持独立。
- 固定种子2026，每类4500训练、500验证，划分索引存入results/split_indices.npz。
- 官方原始数据由torchvision下载。若官方慢，可自行下载完全相同的归档放在data/cifar-10-python.tar.gz；脚本要求MD5为c58f30108f718f92721af3b95e74349a。
- 两模型均从零初始化，固定30轮。验证准确率最高的权重用于测试；所有模型训练完成后才评估测试集。
- 数据归一化采用固定mean=std=0.5，不从验证或测试数据计算统计量。
- 输入以uint8缓存在GPU以减少Windows数据加载开销。内存不足时可减小batch；无GPU会回退CPU，但运行时间较长。

## 模型与对照

基础模型：三个双卷积块，通道32/64/128；每块两次Conv3x3+BN，ReLU，MaxPool2；最后全局平均池化、Dropout0.2、线性分类器。
改进模型：增加投影残差、SE通道注意力、反射填充随机裁剪、水平翻转和0.1标签平滑。
两模型均采用AdamW（lr0.002，weight_decay0.0005）及余弦学习率调度（最低0.00002）。这是组合改进对照，不能据此判断单项技术贡献。
训练accuracy包含随机增强和Dropout的影响；验证不增强。改进训练loss含标签平滑，验证loss为普通交叉熵，不能将二者视为相同目标下的loss。

## 文件

- train.py：模型、数据准备、训练、验证和最终测试。
- checkpoints：两个最佳验证checkpoint，保存模型类型、最佳epoch和权重。
- results：每轮日志、固定划分、逐样本测试预测、总体与每类准确率。
- make_figures.py：loss/accuracy曲线、混淆矩阵、每类第一个误分类样例、Grad-CAM。
- verify_results.py：从预测独立重算指标，检查分割和最佳epoch选择。
- build_report.py：从实测结果创建不超过5页的报告，无结果时不会生成虚构数字。
- report：PDF与Markdown报告。

## 单独评估与解释

最佳模型根据validation_accuracy选择，不根据test_accuracy选择。预测保存在NPZ中，包含真实标签、预测标签和10类概率；每类准确率定义为该类正确数除以该类样本数，即召回率。
Grad-CAM针对模型实际预测的类别，在最后一个4×4特征图上计算，再放大显示，不是分割结果。每张热图单独归一化，不能跨图片比较绝对强度。

## 复现限制

固定随机种子能固定数据划分；GPU算子、驱动和版本变化仍可能导致数值差异。本实验仅一个随机种子，不报告统计显著性。实际训练日期与环境、指标以results/summary.json和训练日志为准。不要根据测试集结果反复选择模型或超参数。

数据来源：https://www.cs.toronto.edu/~kriz/cifar.html 。CIFAR数据集原始说明：Alex Krizhevsky, Learning Multiple Layers of Features from Tiny Images, 2009。

直接复算已保存权重：python evaluate.py。完成验证与报告后运行 python package_submission.py 生成提交ZIP，包含权重和结果，原始数据不重复打包。
