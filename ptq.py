# ptq.py
# Post-Training Static Quantization (PTQ): FP32 -> INT8
#
# Steps:
#   1. load trained FP32 model
#   2. fuse conv+bn+relu layers
#   3. set quantization backend + qconfig
#   4. insert observers and calibrate on a subset of training images
#   5. convert to INT8
#   6. save the INT8 model (TorchScript so it runs without this code) and test it
#
# usage:  python ptq.py --calib 2000

import argparse
import time
import torch

from common import (load_fp32, get_calib_loader, get_test_loader, evaluate,
                    file_size_mb, FP32_PATH, PTQ_PATH, QBACKEND)
from calibration import build_ptq_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib", type=int, default=2000, help="number of training images for calibration")
    args = parser.parse_args()

    print("Quantization backend:", QBACKEND)
    fp32 = load_fp32()
    test_loader = get_test_loader("cpu")

    print(f"Building PTQ model (calibration images = {args.calib}) ...")
    t0 = time.time()
    int8_model = build_ptq_model(fp32, get_calib_loader(args.calib))
    print(f"Done in {time.time() - t0:.1f}s")

    # show a small part of the converted model so we can see INT8 layers
    print("\nConverted first block:")
    print(int8_model.layer1[0])

    # save as TorchScript -> "executable" checkpoint, can be loaded with torch.jit.load
    scripted = torch.jit.script(int8_model)
    torch.jit.save(scripted, PTQ_PATH)
    print("\nSaved INT8 PTQ model ->", PTQ_PATH)

    # deploy check: load the saved file again and run on the test set
    loaded = torch.jit.load(PTQ_PATH)
    fp32_acc = evaluate(fp32, test_loader)
    int8_acc = evaluate(loaded, test_loader)
    print(f"\nFP32 test acc : {fp32_acc:.2f}%   size {file_size_mb(FP32_PATH):.2f} MB")
    print(f"PTQ  test acc : {int8_acc:.2f}%   size {file_size_mb(PTQ_PATH):.2f} MB")
    print(f"Accuracy drop : {fp32_acc - int8_acc:.2f}%")


if __name__ == "__main__":
    main()
