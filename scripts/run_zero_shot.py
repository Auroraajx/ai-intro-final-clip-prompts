"""
run_zero_shot.py — First experiment: zero-shot CLIP on CIFAR-10/100.

Usage:
    python scripts/run_zero_shot.py --dataset cifar10 --templates 5

Output:
    - prints top-1 accuracy
    - saves confusion matrix to results/fig_zero_shot_confusion.png
"""
import argparse
import os
import pathlib

import torch
import open_clip
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm


# 手写 5 个不同风格的模板，后面 RQ1 会扩展到 10/20/40/80
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a low-resolution photo of a {}.",
    "this is a {}.",
    "a photo of the {} taken in the wild.",
]


def build_transform(clip_model):
    """CLIP 官方推荐的预处理：resize + center crop + normalize."""
    img_size = clip_model.visual.image_size[-1]  # ViT-B/32 -> 224
    return Compose([
        Resize(img_size, interpolation=Image.BICUBIC),
        CenterCrop(img_size),
        ToTensor(),
        Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--templates", type=int, default=5,
                        help="用前 N 个模板（后面 RQ1 会扫这个数）")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-root", default="./data")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] using device: {device}")

    # 1. 加载 CLIP（权重会自动下载到 ~/.cache/ ，不需要 OpenAI 账号）
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    # 2. 加载数据集
    if args.dataset == "cifar10":
        ds = CIFAR10(root=args.data_root, train=False, download=True, transform=preprocess)
        classes = CIFAR10.classes
    else:
        ds = CIFAR100(root=args.data_root, train=False, download=True, transform=preprocess)
        classes = CIFAR100.classes

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 3. 把每个类别的所有模板编码成文本特征，再平均 -> 每个类别一个文本向量
    templates = PROMPT_TEMPLATES[: args.templates]
    print(f"[info] using {len(templates)} templates for {len(classes)} classes")

    text_features_all = []
    for class_name in classes:
        texts = tokenizer([t.format(class_name.replace("_", " ")) for t in templates]).to(device)
        feat = model.encode_text(texts)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feat = feat.mean(dim=0)           # 多模板集成：平均
        feat = feat / feat.norm()
        text_features_all.append(feat)
    text_features = torch.stack(text_features_all)  # [num_classes, dim]

    # 4. 遍历测试集，算 image feature 和每个文本 feature 的余弦相似度，取最大
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="evaluating"):
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        sims = image_features @ text_features.T   # [B, num_classes]
        preds = sims.argmax(dim=-1).cpu()

        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    acc = correct / total
    print(f"[result] zero-shot top-1 accuracy on {args.dataset}: {acc:.4f}")

    # 5. 存混淆矩阵图
    out_dir = pathlib.Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=45, values_format="d", colorbar=False)
    plt.title(f"Zero-shot CLIP on {args.dataset} (templates={len(templates)}, acc={acc:.3f})")
    plt.tight_layout()
    out_path = out_dir / "fig_zero_shot_confusion.png"
    plt.savefig(out_path, dpi=150)
    print(f"[info] saved confusion matrix -> {out_path}")


if __name__ == "__main__":
    from PIL import Image  # 延迟导入，避免头部循环
    main()
