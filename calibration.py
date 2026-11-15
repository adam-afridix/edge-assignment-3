# calibration.py
# Calibration pipeline for Post-Training Static Quantization.
#
# The observers inside the prepared model need to "see" some real data so they
# can record the min / max (histogram) of every activation. From these ranges
# the scale and zero-point of each INT8 tensor are computed during convert().
# We use a random subset of the TRAINING set, never the test set.
#
# Running this file directly does a small experiment: it tries different
# calibration sizes and prints the INT8 accuracy for each one.
#
# usage:  python calibration.py

import os
import csv
import copy
import time
import torch
import torch.ao.quantization as tq

from common import (load_fp32, fuse_model, get_calib_loader, get_test_loader,
                    evaluate, QBACKEND, RESULT_DIR)


@torch.no_grad()
def calibrate(prepared_model, calib_loader, verbose=True):
    """Feed calibration batches through the model (observers collect stats)."""
    prepared_model.eval()
    n = 0
    for x, _ in calib_loader:
        prepared_model(x)
        n += x.size(0)
    if verbose:
        print(f"  calibration done on {n} images")


def build_ptq_model(fp32_model, calib_loader, backend=QBACKEND):
    """FP32 model -> fused -> observers inserted -> calibrated -> INT8"""
    torch.backends.quantized.engine = backend
    model = copy.deepcopy(fp32_model).cpu().eval()

    # 1) fusion (conv+bn+relu, conv+bn)
    fuse_model(model, qat=False)

    # 2) backend config. x86 qconfig = per-channel weights + histogram observer for activations
    model.qconfig = tq.get_default_qconfig(backend)

    # 3) insert observers
    tq.prepare(model, inplace=True)

    # 4) calibration
    calibrate(model, calib_loader)

    # 5) convert to INT8
    tq.convert(model, inplace=True)
    return model


def main():
    test_loader = get_test_loader("cpu")
    fp32 = load_fp32()

    rows = []
    for n in [100, 500, 1000, 2000, 5000]:
        print(f"Calibration with {n} images")
        t0 = time.time()
        q = build_ptq_model(fp32, get_calib_loader(n))
        acc = evaluate(q, test_loader)
        print(f"  INT8 test acc = {acc:.2f}%  ({time.time() - t0:.1f}s)")
        rows.append([n, round(acc, 2)])

    out = os.path.join(RESULT_DIR, "ptq_calibration_size.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["calibration_images", "ptq_int8_test_acc"])
        w.writerows(rows)
    print("Saved ->", out)


if __name__ == "__main__":
    main()
