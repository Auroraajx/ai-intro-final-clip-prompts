"""Download CIFAR-10 from hf-mirror (China-friendly) and convert to torchvision format."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time
import pickle
import numpy as np
from datasets import load_dataset

OUT_DIR = "data/cifar-10-batches-py"
os.makedirs(OUT_DIR, exist_ok=True)

print("[1/3] Loading CIFAR-10 from hf-mirror (this downloads parquet)...")
t0 = time.time()
ds = load_dataset("uoft-cs/cifar10")
print(f"    done in {time.time()-t0:.1f}s")

classes = ds["train"].features["label"].names
print(f"    classes: {classes}")

# Split: train 50000 -> 5 batches of 10000; test 10000 -> 1 batch
def to_batch(images, labels):
    data = np.array([np.array(img).reshape(3, 32, 32) for img in images])
    data = data.transpose(0, 1, 3, 2).reshape(len(images), -1)  # to (N, 3072)
    return {
        b"data": data.astype(np.uint8),
        b"labels": list(labels),
    }

print("[2/3] Writing pickle batches...")
train_imgs = ds["train"]["img"]
train_labels = ds["train"]["label"]
for i in range(5):
    start = i * 10000
    end = start + 10000
    batch = to_batch(train_imgs[start:end], train_labels[start:end])
    with open(f"{OUT_DIR}/data_batch_{i+1}", "wb") as f:
        pickle.dump(batch, f)
    print(f"    data_batch_{i+1} written")

test_batch = to_batch(ds["test"]["img"], ds["test"]["label"])
with open(f"{OUT_DIR}/test_batch", "wb") as f:
    pickle.dump(test_batch, f)
print("    test_batch written")

meta = {
    b"label_names": [c.encode() for c in classes],
    b"num_cases_per_batch": 10000,
    b"num_vis": 3072,
}
with open(f"{OUT_DIR}/batches.meta", "wb") as f:
    pickle.dump(meta, f)

# Also write the tar.gz marker so torchvision won't try to re-download
print("[3/3] Done. CIFAR-10 is ready at", OUT_DIR)
