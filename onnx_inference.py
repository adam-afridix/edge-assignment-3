# onnx_inference.py
# CPU inference with ONNX Runtime for the exported models.
# This file does NOT need PyTorch models, only the .onnx files + numpy/PIL.
#
# examples:
#   python onnx_inference.py --model qat_int8 --samples 5        (predict random test images)
#   python onnx_inference.py --model fp32 --image my_dog.jpg      (your own image)
#   python onnx_inference.py --benchmark                          (all models on full test set -> csv)

import argparse
import csv
import os
import time
import numpy as np
import onnxruntime as ort
from PIL import Image

from perf_mode import high_performance

ROOT = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.join(ROOT, "onnx_models")
DATA_DIR = os.path.join(ROOT, "data")
RESULT_DIR = os.path.join(ROOT, "results")
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
MODELS = ["fp32", "ptq_int8", "qat_int8", "unstructured_pruned", "structured_pruned"]


def normalize(imgs_uint8):
    """(N,32,32,3) uint8 -> (N,3,32,32) float32 normalized"""
    x = imgs_uint8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(0, 3, 1, 2).copy()


def load_image(path):
    img = Image.open(path).convert("RGB").resize((32, 32))
    return normalize(np.array(img)[None])


def load_session(name, threads=None):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(os.path.join(ONNX_DIR, f"{name}.onnx"), opts,
                                providers=["CPUExecutionProvider"])


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def benchmark(threads=None):
    d = np.load(os.path.join(DATA_DIR, "cifar10_test.npz"))
    x_all, y_all = normalize(d["x"]), d["y"]
    rows = []
    for name in MODELS:
        sess = load_session(name, threads)
        # accuracy + throughput, batch 256
        preds = []
        t0 = time.perf_counter()
        for i in range(0, len(x_all), 256):
            preds.append(sess.run(None, {"input": x_all[i:i + 256]})[0].argmax(1))
        thr = len(x_all) / (time.perf_counter() - t0)
        acc = 100 * (np.concatenate(preds) == y_all).mean()
        # latency, batch 1
        one = x_all[:1]
        for _ in range(30):
            sess.run(None, {"input": one})
        times = []
        for _ in range(500):
            t0 = time.perf_counter()
            sess.run(None, {"input": one})
            times.append((time.perf_counter() - t0) * 1000)
        size = os.path.getsize(os.path.join(ONNX_DIR, f"{name}.onnx")) / 2 ** 20
        rows.append({"model": name, "test_accuracy_%": round(acc, 2), "onnx_size_MB": round(size, 3),
                     "latency_median_ms_bs1": round(float(np.median(times)), 3),
                     "latency_p95_ms_bs1": round(float(np.percentile(times, 95)), 3),
                     "throughput_img_per_s_bs256": round(thr, 1)})
        print(f"{name:<22} acc {acc:6.2f}% | {size:6.2f} MB | {np.median(times):6.3f} ms/img | {thr:8.0f} img/s")

    out = os.path.join(RESULT_DIR, "onnx_benchmark.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved ->", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=MODELS + ["all"])
    parser.add_argument("--image", nargs="*", default=None)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    print("ONNX Runtime", ort.__version__, "| high performance mode:", high_performance())
    if args.benchmark:
        benchmark(args.threads)
        return

    if args.image:
        x = np.concatenate([load_image(p) for p in args.image])
        truth, ids = ["?"] * len(x), [os.path.basename(p) for p in args.image]
    else:
        d = np.load(os.path.join(DATA_DIR, "cifar10_test.npz"))
        idx = np.random.default_rng(7).choice(len(d["y"]), args.samples, replace=False)
        x = normalize(d["x"][idx])
        truth, ids = [CLASSES[d["y"][i]] for i in idx], [f"test[{i}]" for i in idx]

    names = MODELS if args.model == "all" else [args.model]
    for name in names:
        sess = load_session(name, args.threads)
        t0 = time.perf_counter()
        probs = softmax(sess.run(None, {"input": x})[0])
        ms = (time.perf_counter() - t0) * 1000
        print(f"\n===== {name}.onnx  ({len(x)} images in {ms:.2f} ms) =====")
        for i in range(len(x)):
            p = probs[i].argmax()
            mark = "" if truth[i] == "?" else ("  OK" if CLASSES[p] == truth[i] else "  X")
            print(f"{str(ids[i]):<14} true: {truth[i]:<11} pred: {CLASSES[p]:<11} {probs[i, p]*100:5.1f}%{mark}")


if __name__ == "__main__":
    main()
