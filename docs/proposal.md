# Project Proposal — Can Better Prompts Replace Fine-tuning?

> **课程**：人工智能导论（Introduction to AI）
> **赛道**：Research Track（研究型）
> **题目**：R1 — Can Better Prompts Replace Fine-tuning?
> **作者**：学号 251240110（GitHub: Auroraajx，单人项目）
> **提交节点**：2026-11-16 23:59（Soft DDL）
> **代码仓库**：https://github.com/Auroraajx/ai-intro-final-clip-prompts

---

## 1. 项目背景与问题定义

### 1.1 背景

CLIP（Radford et al., 2021）通过在约 4 亿图文对上做对比学习，获得了强大的零样本（zero-shot）迁移能力：不经过任何任务训练，即可按文本类别对图像分类。但零样本性能高度依赖**手工提示词模板**（prompt template）的设计——同一个类别名，套进不同句式（"a photo of a {}." 与 "this is a {}."），结果可能明显不同。CoOp（Zhou et al., 2022）进一步表明，将提示词上下文变为**可学习向量**，在少样本场景下能大幅逼近全参数微调。

### 1.2 研究问题与"替代 fine-tuning"的界定

本项目要回答一个经验性问题：

> **在少样本图像分类场景下，精心设计的手工提示词，能够在多大程度上替代"参数学习"？**

官方题目特别要求明确"替代 fine-tuning"指替代哪一类微调。本项目的界定如下：

- **"被替代的对象"**：全参数监督微调（full fine-tuning，冻结 CLIP 前的设定下对骨干全量训练），以及参数高效的两种中间方案——线性探针（linear probe）和 LoRA（低秩适配）。
- **"替代的手段"**：零训练（zero-training）的手工提示词模板（hand-crafted prompt templates），含多模板集成与类别描述粒度调整。
- 注意：本研究**不涉及**可学习提示词（如 CoOp 式 prompt learning），因为那本身也是一种参数学习；我们聚焦纯手工提示词，从而把"写得好不好"与"训不训参数"这两个因素彻底分开。

**核心研究问题**：给定相同的少样本标注预算，手工提示词工程相比线性探针、LoRA、全参数微调，各自的准确率 / 训练成本 / 推理成本是多少？提示词的优势在什么条件下消失？

## 2. 相关工作

- **CLIP（Radford et al., ICML 2021）**：对比语言-图像预训练。其核心贡献之一是零样本分类范式，论文使用 80 个手工模板集成作为默认设置，也是本项目 baseline 的直接来源。
- **CoOp（Zhou et al., IJCV 2022）**：将提示词上下文词改为可学习向量，在 few-shot 下显著超过手工提示词，逼近全参微调。CoOp 是"提示词即参数"的代表，本研究以它为参照系划清边界：我们只研究手工提示词，不研究可学习提示词。
- **LoRA（Hu et al., ICLR 2022）**：冻结预训练权重，用两个低秩矩阵表示权重更新。作为参数高效微调（PEFT）的代表纳入对比，量化"极小参数量下的性能上限"。
- **Linear Probe**：冻结骨干特征、训练线性分类头，是 CLIP 原文与下游评测中的标准弱微调方案，作为"最便宜的参数学习"参与对比。

## 3. 方法与实验设计

### 3.1 数据集与模型

| 项目 | 选择 | 说明 |
|---|---|---|
| 数据集 | CIFAR-10、CIFAR-100 | 均 32×32 小图；从 Hugging Face 镜像下载，已缓存本地 |
| 骨干模型 | OpenAI CLIP `ViT-B/32`（冻结） | 预训练权重，本地离线推理，无需 API Key |
| 评价指标 | Top-1 准确率（主指标）；可训练参数量、显存峰值、单 epoch 时长（效率指标） | 测试集仅用于最终评估一次 |

### 3.2 Baseline 与对比方法

1. **Zero-shot 手工提示词**：模板数 ∈ {5, 10, 20, 40, 80}，多模板特征平均集成（对齐 CLIP 原文 80 模板设置）。
2. **Linear Probe**：冻结 CLIP 特征，训练线性分类头。
3. **LoRA**：作用于视觉 Transformer，rank ∈ {2, 4, 8, 16}。
4. **Full Fine-tuning（可选）**：小学习率全参微调，作为性能上界参考。

### 3.3 研究问题与变量控制

- **RQ1 模板数量**：模板数对 zero-shot 准确率的影响曲线（边际收益是否在 ~10 模板后趋缓）。
- **RQ2 描述粒度**：类别描述从单词 → 短句 → 详细句，对 zero-shot 的影响。
- **RQ3 少样本权衡**：每类 1/2/4/8/16 样本下，手工提示词 vs Linear Probe vs LoRA vs 全参微调的性价比。
- **RQ4 错误分析**：混淆矩阵 + t-SNE，分析 CLIP 在语义相近类别（猫/狗、汽车/卡车等）上的系统性混淆。

**公平对比设置**：所有方法共用同一数据划分（固定随机种子 42 划分训练/验证/测试）；少样本子集一次抽取、全方法共用；训练预算一致（相同 epoch 数与学习率调度）；测试集仅最终评估一次，不用于调参。

### 3.4 预期实验表格与图表

| 编号 | 内容 | 形式 |
|---|---|---|
| 表 1 | 主结果：四种方法 × CIFAR-10/100，准确率 / 参数量 / 显存 / 时间 | Markdown 表格（README 内） |
| 图 1 | 模板数量 vs 准确率 | 折线图 |
| 图 2 | 描述粒度对比 | 柱状图 |
| 图 3 | 少样本学习曲线（1–16 shot × 四方法） | 折线图 |
| 图 4 | LoRA rank 的准确率-参数量 Pareto | 散点图 |
| 图 5 | 混淆矩阵 + t-SNE | 热力图 / 散点图 |

## 4. 个人时间安排（对齐课程时间线）

| 时间 | 阶段 | 本项目的具体任务 |
|---|---|---|
| 11.3–11.9 | 个人选题 | 确定 Research Track + R1；Track/题目登记已并入本次 Proposal 提交 |
| 11.10–11.16 | **Proposal** | 撰写并提交本提案（**11.16 23:59**） |
| 11.17–11.23 | Baseline | 完成 zero-shot 模板扫描（5/10/20/40/80）与 RQ1、RQ2 首轮实验 |
| 11.24–11.30 | **Milestone** | 提交 2–3 页报告 + baseline 结果图（**11.30 23:59**） |
| 12.1–12.7 | 实验改进 | Linear Probe、LoRA（rank 2/4/8/16）、全参微调 |
| 12.8–12.14 | 范围冻结 | 补 RQ4 错误分析（混淆矩阵/t-SNE）；冻结实验范围 |
| 12.15–12.21 | 路演准备 | 整理代码与图表；制作 Slides；录 Demo；提交材料（**12.21 23:59**） |
| 12.22–12.28 | 路演与归档 | 4 分钟路演；提交 Final Report PDF、源码、Demo Video（**12.28 23:59**） |

## 5. 预期成果

1. 公开 GitHub 仓库：可复现脚本 + 结果图 + README 汇总表。
2. 4–6 张实验图表 + 1 张主结果表（见 3.4）。
3. 4–6 页 Final Report（PDF）与 10 分论文。
4. 路演 Slides + 短 Demo 视频（可现场展示实时 zero-shot 分类）。

## 6. 计算资源需求

- **主要平台**：本地 Windows 笔记本（NVIDIA GPU，CUDA 可用）。CLIP ViT-B/32 冻结推理与 Linear Probe、LoRA 训练显存需求均 < 4GB，本地可完全承担。
- **备用方案**：若全参微调显存不足，降级为「LoRA + 冻结部分层」；仍不足则迁移至 Kaggle 免费 T4（16GB）。
- **网络依赖**：数据集与权重均已缓存本地，实验阶段离线可跑；在线资源仅剩 GitHub 同步与最终提交。

## 7. 主要风险与备用方案

| 风险 | 影响 | 应对 |
|---|---|---|
| 全参微调显存不足/训练慢 | RQ3 对比缺上界 | 用 LoRA rank=16 替代上界；必要时 Kaggle T4 |
| 国外数据源下载慢 | 环境阻塞 | 已通过 Hugging Face 国内镜像解决并缓存 |
| 测试集反复调参 | 结果虚高 | 测试集只评估一次；调参只用训练/验证划分 |
| 实验范围失控 | 七周内做不完 | 第 6 周强制范围冻结；RQ3/RQ4 为必做，其余按优先级裁剪 |

## 8. 参考文献

1. Radford A, Kim J W, Hallacy C, et al. Learning Transferable Visual Models From Natural Language Supervision[C]. ICML, 2021. (arXiv:2103.00020)
2. Zhou K, Yang J, Loy C C, et al. Learning to Prompt for Vision-Language Models[J]. IJCV, 2022. (arXiv:2109.01134)
3. Hu E J, Shen Y, Wallis P, et al. LoRA: Low-Rank Adaptation of Large Language Models[C]. ICLR, 2022. (arXiv:2106.09685)
