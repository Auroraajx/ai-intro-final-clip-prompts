# 必读论文阅读提纲（R1）

> 用于 AI 导论 Final Project R1，配套仓库 docs/proposal.md。
> 阅读产出将直接变成 proposal §2 相关工作 + Final Report Related Work 的底稿。

## 速览：三篇论文是什么、在我们项目里是谁

### 一句话摘要

| 论文 | 一句话 | 要不要训参数 | 项目身份 |
|---|---|---|---|
| **CLIP** | 图文对比预训练，提供 zero-shot 分类能力 | 预训练已完成，推理不再训 | 骨干模型（baseline 本体） |
| **CoOp** | 提示词上下文变成可学习向量 | 要训（提示词参数） | 边界参照：我们不做这一类 |
| **LoRA** | 冻结主干 + 两个低秩小矩阵 | 训极少参数 | RQ3 对比方法之一 |

### 方法谱系（按"训不训参数、训多少"分层）

```
零训练（不更新任何参数）
  └─ 手工模板提示词              ← RQ1 / RQ2 做的事
训少量参数
  ├─ Linear Probe（只训线性分类头） ← RQ3 对比
  └─ LoRA（低秩小矩阵）           ← RQ3 对比
训提示词参数
  └─ CoOp（可学习上下文向量）       ← 边界参照，不做
全参数训练
  └─ Full Fine-tuning（性能上界）  ← RQ3 对比（可选）
```

> 整个 R1 沿这条谱系回答：**纯手工提示词**在哪个档位、花多少标注预算，能追上上面哪一档参数学习方法。

## 0. 通用笔记模板（每篇论文按这个记）

```
## 论文名（作者, 年份, 会议/期刊）
- [ ] 一句话贡献：________
- [ ] 方法一句话：________
- [ ] 关键图/表（论文里最说明问题的一张）：________
- [ ] 核心结论：________
- [ ] 局限/没说清的地方：________
- [ ] 和 R1 的关系：________（引用它的哪个设定？我们对比谁？）
- [ ] 值得抄的实验设置：________（如数据集、指标、对比表结构）
```

---

## P0-1：CoOp（官方指定必读）

**Learning to Prompt for Vision-Language Models** — Zhou et al., IJCV 2022
https://arxiv.org/abs/2109.01134

### 读之前先想清楚的问题
1. 手工 prompt 和可学习 prompt 的区别到底在哪？（一句话）
2. CoOp 声称逼近全参微调，它用什么实验证明的？（few-shot 多少样本？）
3. 为什么说"prompt learning 本身也是参数学习"？

### 精读清单（按顺序）
- [ ] **摘要 + Intro**：他们解决的问题、核心做法、宣称的增益
- [ ] **方法 3.1（Context Optimization）**：上下文向量怎么初始化、怎么优化；读出"它训练的参数是 prompt 向量，不是主干权重"这一点
- [ ] **实验对比表**：重点看 few-shot（1/2/4/8/16 shot）下 CoOp vs 手工 prompt vs 微调的数字
- [ ] **与手工 prompt 的对比**：CoOp 比手工模板强多少？
- [ ] **与全参微调/Linear Probe 的对比**：差距还有多大？在什么数据集上接近？

### 笔记必答项
- CoOp 用的评估数据集有哪些？（我们是否该学它用多个数据集？）
- CoOp 的基线里，手工 prompt 用的是几个模板？（对比我们 RQ1 的 80 模板设定）

### 对我们的直接影响
- **RQ3 的对比框架直接抄它**：same dataset, same few-shot shots, 比 CoOp / 手工 prompt / Linear Probe / 微调
- 我们和它划清边界：**我们不训 prompt 向量，只用手工模板** → 我们回答的是"纯手工提示词值多少"

---

## P0-2：CLIP（实验必需）

**Learning Transferable Visual Models From Natural Language Supervision** — Radford et al., ICML 2021
https://arxiv.org/abs/2103.00020

### 读之前先想清楚的问题
1. zero-shot 分类到底怎么运作的？（图文向量 → 余弦相似度 → argmax）
2. 80 模板集成为什么比 1 个模板准？（多个句式的特征平均，抵消句式偏差）
3. 对比学习训练的目标是什么？（拉近匹配图文对，推远不匹配）

### 精读清单（按顺序）
- [ ] **摘要 + Intro**：4 亿图文对、对比学习、zero-shot 能力
- [ ] **方法（训练目标）**：InfoNCE 对比损失长什么样（不用背公式，理解"拉近/推远"即可）
- [ ] **3.6.3 Prompt Engineering**：模板怎么设计、为什么用 80 个模板集成（**RQ1 的源头**）
- [ ] **Table 2（zero-shot 结果）**：CLIP 在标准数据集上的 zero-shot 表现，找 CIFAR-10/100 那一行

### 笔记必答项
- 80 个模板里大概有几类句式？（光线/画质/尺寸/背景……）——抄作业时参考
- CIFAR-10 / CIFAR-100 上 CLIP ViT-B/32 的 zero-shot 官方数字是多少？（我们 85.8% 要和它对表）

### 对我们的直接影响
- 骨干模型：OpenAI CLIP ViT-B/32（已加载）
- RQ1 的 80 模板 = 复现 CLIP 原文设置
- RQ2 描述粒度 = 扩展它的 prompt engineering 讨论

---

## P1：LoRA（RQ3 前必读）

**LoRA: Low-Rank Adaptation of Large Language Models** — Hu et al., ICLR 2022
https://arxiv.org/abs/2106.09685

### 读之前先想清楚的问题
1. 为什么冻结权重 + 两个小矩阵 ≈ 更新整个大矩阵？（权重更新的低秩近似）
2. rank 是什么？rank=2 和 rank=16 差在哪？（参数量、表达能力）
3. 为什么 LoRA 显存省？（不用给主干权重存优化器状态）

### 精读清单（按顺序）
- [ ] **摘要 + Intro**：参数高效微调的动机
- [ ] **方法（Figure 1 / 公式 1-3）**：W + BA 的结构，A 随机初始化、B 零初始化
- [ ] **实验：rank 的影响**（r 从 1/2/4/8/16 变时，效果/参数量怎么变）
- [ ] **效率对比**：训练显存、训练速度 vs 全参微调

### 笔记必答项
- LoRA 一般加在哪些层？（Transformer 的 Q/K/V/O 投影？）
- rank 增大到多少之后收益趋平？（我们 RQ3 扫 2/4/8/16 的依据）

### 对我们的直接影响
- RQ3 的对比方法之一：在 CLIP 视觉 Transformer 上加 LoRA，扫 rank ∈ {2,4,8,16}
- 参数量/显存/时间是我们要报告的效率指标

---

## 阅读节奏建议

| 论文 | 建议时间 | 产出 |
|---|---|---|
| CLIP | 半天 | 笔记 + 确认官方 CIFAR-10/100 数字 |
| CoOp | 半天 | 笔记 + RQ3 对比框架定稿 |
| LoRA | 2 小时 | 笔记 + RQ3 的 rank 档位定稿 |

> 每篇读完把笔记贴到仓库 `docs/`（新建 `notes_clip.md`、`notes_coop.md`、`notes_lora.md`），
> Final Report 的 Related Work 直接从笔记里提出来写。
