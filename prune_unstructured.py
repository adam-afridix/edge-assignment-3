# prune_unstructured.py
# Iterative unstructured (magnitude) pruning + fine-tuning.
#
# Idea: weights with the smallest absolute value matter the least, so we set
# them to zero. We do it slowly in a few iterations and fine-tune after each
# one so the network can adjust:
#
#     iteration:   1     2     3     4     5
#     sparsity :  20%   40%   60%   70%   80%   (of all conv + fc weights)
#
# torch.nn.utils.prune adds a "weight_mask" buffer and keeps "weight_orig".
# The real weight is weight_orig * weight_mask (computed in a forward hook).
# At the end prune.remove() makes it permanent: the zeros are written into
# the normal "weight" tensor and the mask is deleted.
#
# usage:  python prune_unstructured.py --targets 0.2 0.4 0.6 0.7 0.8 --ft_epochs 2

import argparse
import csv
import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from common import (load_fp32, get_loaders, evaluate, fine_tune, get_device, set_seed,
                    count_params, file_size_mb, zipped_size_mb,
                    UNSTRUCT_PATH, FP32_PATH, RESULT_DIR)


def prunable_layers(model):
    """all conv and linear layers -> list of (module, 'weight')"""
    return [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def weight_sparsity(model):
    """percentage of zeros in conv + fc weights (works with and without masks)"""
    zeros, total = 0, 0
    for m, _ in prunable_layers(model):
        zeros += (m.weight == 0).sum().item()
        total += m.weight.numel()
    return 100.0 * zeros / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.7, 0.8])
    parser.add_argument("--ft_epochs", type=int, default=2, help="fine-tune epochs after every iteration")
    parser.add_argument("--final_epochs", type=int, default=3, help="extra fine-tune at the end")
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    train_loader, test_loader = get_loaders(128, device)

    model = load_fp32().to(device)
    base_acc = evaluate(model, test_loader, device)
    print(f"Original FP32 accuracy: {base_acc:.2f}%\n")

    params = prunable_layers(model)
    log = [["iteration", "target_sparsity_%", "actual_sparsity_%", "acc_after_prune_%", "acc_after_finetune_%"]]
    log.append([0, 0, round(weight_sparsity(model), 2), round(base_acc, 2), round(base_acc, 2)])

    prev = 0.0
    for it, target in enumerate(args.targets, start=1):
        # amount is taken from the weights that are still alive,
        # so convert the overall target into a step amount
        step = (target - prev) / (1.0 - prev)
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=step)
        prev = target

        acc_pruned = evaluate(model, test_loader, device)
        print(f"Iteration {it}: target {target*100:.0f}% | real sparsity {weight_sparsity(model):.2f}% "
              f"| acc right after pruning {acc_pruned:.2f}%")

        epochs = args.ft_epochs + (args.final_epochs if it == len(args.targets) else 0)
        acc_ft = fine_tune(model, train_loader, test_loader, device, epochs, args.lr, tag=f"iter {it}")
        log.append([it, round(target * 100, 1), round(weight_sparsity(model), 2),
                    round(acc_pruned, 2), round(acc_ft, 2)])
        print()

    # make pruning permanent (remove masks + weight_orig, keep zeros in weight)
    for m, name in params:
        prune.remove(m, name)

    model = model.cpu().eval()
    torch.save(model.state_dict(), UNSTRUCT_PATH)

    # per layer sparsity
    print("Per-layer sparsity after pruning:")
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            s = 100.0 * (m.weight == 0).sum().item() / m.weight.numel()
            print(f"  {n:<28} {tuple(m.weight.shape)!s:<20} {s:6.2f}% zeros")

    total, nz = count_params(model)
    print(f"\nTotal params {total:,} | non-zero {nz:,} | conv/fc weight sparsity {weight_sparsity(model):.2f}%")
    print(f"Checkpoint size {file_size_mb(UNSTRUCT_PATH):.2f} MB (zipped {zipped_size_mb(UNSTRUCT_PATH):.2f} MB)"
          f" vs FP32 {file_size_mb(FP32_PATH):.2f} MB (zipped {zipped_size_mb(FP32_PATH):.2f} MB)")
    print("Saved unstructured pruned model ->", UNSTRUCT_PATH)

    out = os.path.join(RESULT_DIR, "unstructured_iterations.csv")
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(log)
    print("Saved ->", out)


if __name__ == "__main__":
    main()
