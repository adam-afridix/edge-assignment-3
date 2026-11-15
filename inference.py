# inference.py
# Inference pipeline for all five PyTorch models (runs on CPU).
#
#   fp32          - baseline ResNet-18
#   ptq_int8      - post-training static quantized model
#   qat_int8      - quantization-aware trained model
#   unstructured  - magnitude pruned model (80% zeros)
#   structured    - channel pruned and rebuilt model
#
# examples:
#   python inference.py --model qat_int8 --samples 8          (random test images)
#   python inference.py --model all --samples 5
#   python inference.py --model structured --image my_cat.jpg  (your own image)

import argparse
import os
import time
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from common import LOADERS, CLASSES, MEAN, STD, get_test_set
from perf_mode import high_performance


def preprocess(path):
    """any image -> 1x3x32x32 normalized tensor"""
    img = Image.open(path).convert("RGB")
    tf = T.Compose([T.Resize((32, 32)), T.ToTensor(), T.Normalize(MEAN, STD)])
    return tf(img).unsqueeze(0)


@torch.no_grad()
def predict(model, x, topk=3):
    t0 = time.perf_counter()
    logits = model(x)
    ms = (time.perf_counter() - t0) * 1000
    probs = F.softmax(logits, dim=1)
    p, idx = probs.topk(topk, dim=1)
    return p, idx, ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=list(LOADERS.keys()) + ["all"])
    parser.add_argument("--image", nargs="*", default=None, help="path(s) to your own images")
    parser.add_argument("--samples", type=int, default=5, help="random test images if no --image given")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    high_performance()
    names = list(LOADERS.keys()) if args.model == "all" else [args.model]

    # build the input batch
    if args.image:
        x = torch.cat([preprocess(p) for p in args.image])
        truth = ["?"] * len(args.image)
        ids = [os.path.basename(p) for p in args.image]
    else:
        test_set = get_test_set()
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(test_set), generator=g)[:args.samples].tolist()
        x = torch.stack([test_set[i][0] for i in idx])
        truth = [CLASSES[test_set[i][1]] for i in idx]
        ids = [f"test[{i}]" for i in idx]

    for name in names:
        model = LOADERS[name]()
        model(x[:1])   # warm-up
        p, top, ms = predict(model, x)
        print(f"\n===== {name}  (batch of {len(x)} in {ms:.2f} ms) =====")
        print(f"{'image':<14}{'true':<12}{'predicted':<12}{'conf':>7}   top-3")
        for i in range(len(x)):
            top3 = ", ".join(f"{CLASSES[top[i, j]]} {p[i, j]*100:.0f}%" for j in range(3))
            pred = CLASSES[top[i, 0]]
            mark = "" if truth[i] == "?" else ("  OK" if pred == truth[i] else "  X")
            print(f"{str(ids[i]):<14}{truth[i]:<12}{pred:<12}{p[i, 0]*100:>6.1f}%   {top3}{mark}")


if __name__ == "__main__":
    main()
