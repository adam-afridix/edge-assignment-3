# prepare_data.py
# Downloads CIFAR-10 once and saves it as two small numpy files:
#     data/cifar10_train.npz   (50,000 images)
#     data/cifar10_test.npz    (10,000 images)
# All other scripts read these files.
#
# The official server (cs.toronto.edu) was extremely slow for me, so the script
# first tries the Hugging Face copy of the same dataset (uoft-cs/cifar10) and
# only uses the official torchvision download if that fails.
#
# usage:  python prepare_data.py

import io
import os
import urllib.request
import numpy as np
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HF_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{}-00000-of-00001.parquet"


def from_huggingface(split):
    import pandas as pd
    raw = os.path.join(DATA_DIR, "raw", f"{split}.parquet")
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    if not os.path.exists(raw):
        print(f"downloading {split} split from Hugging Face ...")
        urllib.request.urlretrieve(HF_URL.format(split), raw)
    df = pd.read_parquet(raw)
    # each row: img = {"bytes": png bytes}, label = int
    x = np.stack([np.array(Image.open(io.BytesIO(r["bytes"])).convert("RGB")) for r in df["img"]])
    y = df["label"].to_numpy().astype(np.int64)
    return x, y


def from_torchvision(train):
    import torchvision
    ds = torchvision.datasets.CIFAR10(os.path.join(DATA_DIR, "torchvision"), train=train, download=True)
    return ds.data, np.array(ds.targets, dtype=np.int64)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for split in ["train", "test"]:
        out = os.path.join(DATA_DIR, f"cifar10_{split}.npz")
        if os.path.exists(out):
            print(out, "already exists")
            continue
        try:
            x, y = from_huggingface(split)
        except Exception as e:
            print("Hugging Face download failed:", e, "-> using torchvision")
            x, y = from_torchvision(split == "train")
        assert x.shape[1:] == (32, 32, 3)
        np.savez(out, x=x, y=y)
        print(f"{split}: images {x.shape}, labels {np.bincount(y).tolist()} -> {out}")


if __name__ == "__main__":
    main()
