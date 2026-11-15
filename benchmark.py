# benchmark.py
# Automated benchmark for the saved models. Everything runs on CPU so the
# FP32 and INT8 numbers are fair to compare (INT8 kernels only exist on CPU).
#
# For each model it measures:
#   - top-1 accuracy on the full CIFAR-10 test set (10,000 images)
#   - checkpoint size on disk (and zipped size)
#   - total / non-zero parameters (for the float models)
#   - latency for 1 image (median and p95 of 300 runs)
#   - throughput on the test set with batch 256
# and it also saves every prediction of every model.
#
# usage:
#   python benchmark.py --group quant     (FP32, PTQ-INT8, QAT-INT8)  -> Question 1
#   python benchmark.py --group prune     (FP32, unstructured, structured) -> Question 2
#   python benchmark.py --group all       (all five)

import argparse
import csv
import os
import time
import numpy as np
import torch

from common import (LOADERS, MODEL_FILES, get_test_loader, evaluate, count_params,
                    measure_latency, measure_throughput, file_size_mb,
                    zipped_size_mb, RESULT_DIR, CLASSES)
from perf_mode import high_performance

GROUPS = {
    "quant": ["fp32", "ptq_int8", "qat_int8"],
    "prune": ["fp32", "unstructured", "structured"],
    "all": ["fp32", "ptq_int8", "qat_int8", "unstructured", "structured"],
}
CSV_NAMES = {"quant": "q1", "prune": "q2", "all": "all"}


def run_benchmark(names, tag, threads=None):
    # stop Windows from moving this process to the slow cores (see perf_mode.py)
    print("high performance mode:", high_performance())
    if threads:
        torch.set_num_threads(threads)
    print(f"CPU threads used: {torch.get_num_threads()}")

    # test images are already normalized tensors in memory, so the throughput
    # number is only the model time (no image decoding inside the loop)
    loader = get_test_loader("cpu")
    labels = loader.y.numpy()

    summary, all_preds = [], {}
    base_size, base_lat = None, None

    for name in names:
        print(f"\n=== {name} ===")
        model = LOADERS[name]()

        acc, preds = evaluate(model, loader, "cpu", return_preds=True)
        all_preds[name] = preds
        lat_mean, lat_p95 = measure_latency(model)
        thr = measure_throughput(model, loader)

        # parameter count only makes sense for float (non-scripted) models
        if isinstance(model, torch.jit.ScriptModule):
            total_p, nz_p = "", ""
            sparsity = ""
        else:
            total_p, nz_p = count_params(model)
            sparsity = round(100 * (1 - nz_p / total_p), 2)

        size = file_size_mb(MODEL_FILES[name])
        zsize = zipped_size_mb(MODEL_FILES[name])
        if base_size is None:            # first model of the list is the reference (fp32)
            base_size, base_lat = size, lat_mean

        # per-class accuracy, nice for the report
        per_class = [100 * np.mean(preds[labels == c] == c) for c in range(10)]

        row = {
            "model": name,
            "test_accuracy_%": round(acc, 2),
            "size_MB": round(size, 3),
            "zipped_size_MB": round(zsize, 3),
            "compression_x": round(base_size / size, 2),
            "total_params": total_p,
            "nonzero_params": nz_p,
            "sparsity_%": sparsity,
            "latency_median_ms_bs1": round(lat_mean, 3),
            "latency_p95_ms_bs1": round(lat_p95, 3),
            "speedup_x": round(base_lat / lat_mean, 2),
            "throughput_img_per_s_bs256": round(thr, 1),
        }
        for c in range(10):
            row[f"acc_{CLASSES[c]}"] = round(per_class[c], 1)
        summary.append(row)

        print(f"accuracy {acc:.2f}% | size {size:.2f} MB (zip {zsize:.2f}) | "
              f"latency {lat_mean:.2f} ms | throughput {thr:.0f} img/s")

    # ---- summary csv
    sum_path = os.path.join(RESULT_DIR, f"{tag}_benchmark.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # ---- per-image predictions csv
    pred_path = os.path.join(RESULT_DIR, f"{tag}_predictions.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_index", "true_label", "true_class"] + [f"{n}_pred" for n in names]
                   + [f"{n}_correct" for n in names])
        for i in range(len(labels)):
            preds_i = [int(all_preds[n][i]) for n in names]
            w.writerow([i, int(labels[i]), CLASSES[labels[i]]] + preds_i
                       + [int(p == labels[i]) for p in preds_i])

    # ---- print a small table
    print("\n" + "-" * 88)
    print(f"{'model':<14}{'acc %':>8}{'size MB':>10}{'zip MB':>9}{'compr x':>9}"
          f"{'lat ms':>9}{'speedup':>9}{'img/s':>10}")
    print("-" * 88)
    for r in summary:
        print(f"{r['model']:<14}{r['test_accuracy_%']:>8}{r['size_MB']:>10}{r['zipped_size_MB']:>9}"
              f"{r['compression_x']:>9}{r['latency_median_ms_bs1']:>9}{r['speedup_x']:>9}"
              f"{r['throughput_img_per_s_bs256']:>10}")
    print("-" * 88)
    print("Saved ->", sum_path)
    print("Saved ->", pred_path)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=list(GROUPS.keys()), default="all")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    t0 = time.time()
    run_benchmark(GROUPS[args.group], CSV_NAMES[args.group], args.threads)
    print(f"Total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
