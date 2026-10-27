# Can Better Prompts Replace Fine-tuning?

> A course project for **Introduction to AI**, Nanjing University.
> Exploring how prompt engineering on CLIP trades off against linear probing, full fine-tuning, and LoRA.

- **Author**: Auroraajx
- **Track**: Track 1 (Research)
- **Status**: WIP — Proposal stage

---

## TL;DR

- Uses OpenAI CLIP `ViT-B/32` as a frozen vision-language backbone.
- Systematically studies how **prompt templates**, **class name granularity**, and **few-shot sample size** affect downstream accuracy on CIFAR-10 / CIFAR-100.
- Compares against Linear Probe, full fine-tuning, and LoRA (rank ∈ {2, 4, 8, 16}).

> 结果会随实验推进陆续填进下面的表格和 `results/` 目录。

## Main Results

| Method | Trainable Params | CIFAR-10 Acc | CIFAR-100 Acc |
|---|---|---|---|
| Zero-shot CLIP (80 templates) | 0 | — | — |
| Linear Probe | — | — | — |
| Full Fine-tuning | — | — | — |
| LoRA rank=2 | — | — | — |
| LoRA rank=4 | — | — | — |
| LoRA rank=8 | — | — | — |
| LoRA rank=16 | — | — | — |

## Results Gallery

> 跑出来的图会放在这里。

<!--
![templates vs accuracy](results/fig1_templates_vs_acc.png)
![few-shot curve](results/fig3_few_shot_curve.png)
-->

## Quick Start

```bash
# 1. 建议新建虚拟环境（Python 3.10+）
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. 装依赖
pip install -r requirements.txt

# 3. 跑第一次 zero-shot 实验（会自动下载 CLIP 权重和 CIFAR 数据集）
python scripts/run_zero_shot.py --dataset cifar10 --templates 5
```

**不需要 OpenAI API Key**：CLIP 权重从公开源下载到本地，全程离线推理。

## Project Structure

```
├── docs/               # 课程要求提交的文档
│   ├── proposal.md          # W1-2: Project Proposal (1-2 页)
│   ├── milestone.md         # W3-4: Milestone Report (2-3 页)
│   ├── final_report.md      # W7: Final Report (4-6 页)
│   └── paper.md             # 10 分学术论文版本
├── src/
│   ├── data/             # 数据加载
│   ├── models/           # CLIP 包装、Linear Probe、LoRA
│   ├── prompts/          # 提示词模板库
│   ├── experiments/       # 每个 RQ 一个实验脚本
│   └── utils/
├── scripts/            # 命令行入口
├── results/            # 实验图、表
├── assets/             # README 里嵌的图、demo gif
└── website/            # GitHub Pages 展示页（可选）
```

## Research Questions

- **RQ1**: How does the number / diversity of prompt templates affect zero-shot accuracy?
- **RQ2**: Does finer-grained class description (word → short phrase → detailed sentence) help?
- **RQ3**: In the few-shot regime (1/2/4/8/16 shot), how does "good prompts + linear probe" compare to full fine-tuning and LoRA?
- **RQ4**: Are zero-shot errors concentrated on semantically confusable classes? (confusion matrix + t-SNE)

## Timeline

| Week | Milestone |
|---|---|
| W1 | Scaffold repo, run first zero-shot baseline |
| W2 | Submit Proposal |
| W3 | RQ1 / RQ2 experiments |
| W4 | Milestone Report + Prototype Demo |
| W5 | Linear Probe + full fine-tuning baseline |
| W6 | LoRA sweep + error analysis |
| W7 | Final Report + source code + demo video |

## References

- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), 2021.
- Zhou et al., *Learning to Prompt for Vision-Language Models* (CoOp), 2022.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.

## License

MIT
