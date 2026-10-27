"""
run_zero_shot.py — First experiment: zero-shot CLIP on CIFAR-10.

Usage:
    python scripts/run_zero_shot.py --dataset cifar10 --templates 5

Output:
    - prints top-1 accuracy
    - saves confusion matrix to results/fig_zero_shot_confusion.png
"""
import argparse
import os
import pathlib

# 用国内镜像下载（如果本地已有缓存则直接读）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import open_clip
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import DataLoader, Dataset
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


class HfCifarDataset(Dataset):
    """把 HuggingFace datasets 包装成 torch 能用的 Dataset。"""
    def __init__(self, hf_split, transform):
        self.imgs = hf_split["img"]
        self.labels = hf_split["label"]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx].convert("RGB")
        return self.transform(img), self.labels[idx]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10"], default="cifar10")
    parser.add_argument("--templates", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] using device: {device}")

    # 1. 加载 CLIP
    print("[info] loading CLIP ViT-B/32 ...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    # 2. 加载数据集（从 hf-mirror，已缓存则秒读）
    print("[info] loading CIFAR-10 ...")
    hf_ds = load_dataset("uoft-cs/cifar10")
    classes = hf_ds["train"].features["label"].names
    test_ds = HfCifarDataset(hf_ds["test"], preprocess)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 3. 编码每个类别的所有模板文本，再平均
    templates = PROMPT_TEMPLATES[: args.templates]
    print(f"[info] using {len(templates)} templates for {len(classes)} classes")

    text_features_all = []
    for class_name in classes:
        texts = tokenizer([t.format(class_name.replace("_", " ")) for t in templates]).to(device)
        feat = model.encode_text(texts)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feat = feat.mean(dim=0)
        feat = feat / feat.norm()
        text_features_all.append(feat)
    text_features = torch.stack(text_features_all)  # [num_classes, dim]

    # 4. 遍历测试集
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="evaluating"):
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        sims = image_features @ text_features.T
        preds = sims.argmax(dim=-1).cpu()

        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    acc = correct / total
    print(f"[result] zero-shot top-1 accuracy on {args.dataset}: {acc:.4f}")

    # 5. 存混淆矩阵
    out_dir = pathlib.Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=45, values_format="d", colorbar=False)
    plt.title(f"Zero-shot CLIP on CIFAR-10 (templates={len(templates)}, acc={acc:.3f})")
    plt.tight_layout()
    out_path = out_dir / "fig_zero_shot_confusion.png"
    plt.savefig(out_path, dpi=150)
    print(f"[info] saved confusion matrix -> {out_path}")


if __name__ == "__main__":
    main()
