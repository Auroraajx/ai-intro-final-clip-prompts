# Project Proposal — Can Better Prompts Replace Fine-tuning?

> Course: Introduction to AI
> Track: Track 1 (Research)
> Author: Auroraajx
> Due: Week 2

## 1. Background

Contrastive Language-Image Pre-training (CLIP, Radford et al. 2021) learns a shared
embedding space for images and text on 400M web image–text pairs. At inference time,
it can classify an image into **any** textual category without task-specific training
— a paradigm known as **zero-shot classification**.

However, zero-shot accuracy is highly sensitive to how the user writes the class
name into a prompt template. CoOp (Zhou et al. 2022) further shows that *learnable*
prompts can close much of the gap between zero-shot CLIP and fully fine-tuned models.
This course project asks a simpler, empirical question at undergraduate scale:

> **How much does hand-crafted prompt design actually buy us, and when does it stop
> being worth it compared to actually training some parameters?**

## 2. Research Questions

- **RQ1 — Templates**: How does the number / diversity of prompt templates affect
  zero-shot top-1 accuracy? Does ensembling 80 templates (the CLIP paper setting)
  give a real lift over 5 templates, or is the marginal gain small after ~10?
- **RQ2 — Granularity**: Does describing a class as a single word ("dog"), a short
  phrase ("a photo of a dog"), or a detailed sentence ("a photo of a dog sitting on
  grass, taken in daylight") change zero-shot performance?
- **RQ3 — Few-shot trade-off**: Given a tiny labeled set (1 / 2 / 4 / 8 / 16 shots
  per class), how do three strategies compare?
  (a) well-tuned prompts, no training;
  (b) linear probe on frozen CLIP features;
  (c) low-rank adaptation (LoRA, rank ∈ {2, 4, 8, 16}).
- **RQ4 — Error analysis**: Which classes does zero-shot CLIP systematically
  confuse? Does a confusion matrix / t-SNE of image embeddings reveal a predictable
  failure mode?

## 3. Data & Model

- **Datasets**: CIFAR-10 and CIFAR-100 (auto-downloaded via `torchvision`).
- **Backbone**: OpenAI CLIP `ViT-B/32`, frozen weights from `open_clip_torch`.
  No API key required; weights are downloaded once to local disk.
- **Baselines**:
  1. Zero-shot with 5 / 10 / 20 / 40 / 80 prompt templates.
  2. Linear probe (frozen features, train a linear classifier).
  3. LoRA on the visual transformer, ranks swept.
  4. (Optional) full fine-tuning with a small learning rate.

## 4. Evaluation

- **Primary**: top-1 accuracy on CIFAR-10 / CIFAR-100 test set.
- **Efficiency**: trainable parameter count, GPU memory peak, wall-clock time.
- **Analysis**: confusion matrix, t-SNE of test embeddings.

## 5. Expected Outcomes

- A clean GitHub repository with reproducible scripts.
- 4–6 plots: template count vs accuracy, granularity bar chart, few-shot learning
  curves, LoRA rank Pareto, confusion matrix, t-SNE.
- A 4–6 page Final Report and a separate 10-point academic paper.
- A short demo video (or a live GitHub Pages page) for the 4-minute presentation.

## 6. Timeline

| Week | Plan |
|---|---|
| W1 | Repo scaffold; first zero-shot run on CIFAR-10 with 5 templates. |
| W2 | Write & submit this proposal. |
| W3 | RQ1 (template sweep) and RQ2 (granularity). |
| W4 | Milestone Report + working prototype demo. |
| W5 | Linear probe baseline; start LoRA sweep. |
| W6 | Finish LoRA sweep; confusion matrix + t-SNE. |
| W7 | Final Report, paper, demo video. |
