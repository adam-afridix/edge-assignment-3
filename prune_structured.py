# prune_structured.py
# Structured channel pruning + rebuilding a smaller ResNet-18 + fine-tuning.
#
# Which channels do we remove?
#   Inside every BasicBlock we have   x -> conv1 -> bn1 -> relu -> conv2 -> bn2 -> (+ shortcut)
#   The output channels of conv1 ("mid" channels) are only used by conv2, so we can
#   remove some of them safely. The output of conv2 is added to the shortcut,
#   so those channels are tied to other blocks and we do NOT touch them.
#
# Importance of a channel = L1 norm of its conv1 filter (sum of |w|).
# Channels with the smallest L1 norm are removed.
#
# Keeping the tensor shapes valid when we remove channel set "drop":
#   conv1.weight : [mid, in, 3, 3]  -> keep rows   [keep, :, :, :]
#   bn1 (weight, bias, running_mean, running_var) -> keep [keep]
#   conv2.weight : [out, mid, 3, 3] -> keep columns [:, keep, :, :]
#
# Then a NEW ResNet18 with the smaller "mids" is created and the kept weights
# are copied into it. So the final model is really smaller, there are no
# zero channels or masks left in it.
#
# usage:  python prune_structured.py --ratio 0.5 --ft_epochs 10

import argparse
import torch
import torch.nn as nn

from common import (ResNet18, load_fp32, get_loaders, evaluate, fine_tune, get_device,
                    set_seed, count_params, file_size_mb, STRUCT_PATH, FP32_PATH)


def channel_importance(conv):
    """L1 norm of every output filter of a conv layer"""
    return conv.weight.detach().abs().sum(dim=(1, 2, 3))


def select_channels(model, ratio, multiple_of=8):
    """for every block return the indexes of the mid channels we keep"""
    keep_idx = []
    for blk in model.blocks():
        imp = channel_importance(blk.conv1)
        n = imp.numel()
        n_keep = int(round(n * (1 - ratio) / multiple_of)) * multiple_of   # nice numbers for CPU kernels
        n_keep = max(multiple_of, n_keep)
        idx = torch.argsort(imp, descending=True)[:n_keep]
        keep_idx.append(torch.sort(idx).values)   # keep original order
    return keep_idx


@torch.no_grad()
def build_pruned_model(old, keep_idx):
    """create the smaller network and copy the kept weights into it"""
    new_mids = [len(k) for k in keep_idx]
    new = ResNet18(mids=new_mids)

    # 1) copy everything that has the same shape (stem, shortcuts, bn2, conv2 rows, fc ...)
    new_sd = new.state_dict()
    old_sd = old.state_dict()
    for k in new_sd:
        if new_sd[k].shape == old_sd[k].shape:
            new_sd[k].copy_(old_sd[k])
    new.load_state_dict(new_sd)

    # 2) slice the layers that changed shape
    for ob, nb, keep in zip(old.blocks(), new.blocks(), keep_idx):
        nb.conv1.weight.copy_(ob.conv1.weight[keep])
        nb.bn1.weight.copy_(ob.bn1.weight[keep])
        nb.bn1.bias.copy_(ob.bn1.bias[keep])
        nb.bn1.running_mean.copy_(ob.bn1.running_mean[keep])
        nb.bn1.running_var.copy_(ob.bn1.running_var[keep])
        nb.conv2.weight.copy_(ob.conv2.weight[:, keep])
    return new, new_mids


def count_macs(model):
    """multiply-accumulate operations for one 32x32 image (conv + linear only)"""
    macs = [0]

    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            macs[0] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            macs[0] += m.in_features * m.out_features

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 3, 32, 32))
    for h in hs:
        h.remove()
    return macs[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=0.5, help="fraction of mid channels to remove in every block")
    parser.add_argument("--ft_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.02)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    train_loader, test_loader = get_loaders(128, device)

    old = load_fp32()
    old_acc = evaluate(old.to(device), test_loader, device)
    old.cpu()

    keep_idx = select_channels(old, args.ratio)
    new, new_mids = build_pruned_model(old, keep_idx)

    print("Channel config (mid channels of each block):")
    print("  before:", old.mids)
    print("  after :", new_mids)

    # sanity check: the new network runs and gives the right output shape
    out = new(torch.randn(2, 3, 32, 32))
    print("  forward check output shape:", tuple(out.shape))

    p_old, _ = count_params(old)
    p_new, _ = count_params(new)
    m_old, m_new = count_macs(old), count_macs(new)
    print(f"\nParams: {p_old:,} -> {p_new:,}  ({100 * (1 - p_new / p_old):.1f}% less)")
    print(f"MACs  : {m_old / 1e6:.1f}M -> {m_new / 1e6:.1f}M  ({100 * (1 - m_new / m_old):.1f}% less)")

    new = new.to(device)
    acc_before = evaluate(new, test_loader, device)
    print(f"\nAccuracy: original {old_acc:.2f}% | after channel pruning (no fine-tune) {acc_before:.2f}%")

    print("\nFine-tuning the pruned network ...")
    acc_after = fine_tune(new, train_loader, test_loader, device, args.ft_epochs, args.lr, tag="struct")

    new = new.cpu().eval()
    # save the new channel config together with the weights so the model can be rebuilt
    torch.save({"mids": new_mids, "state_dict": new.state_dict(),
                "prune_ratio": args.ratio}, STRUCT_PATH)
    print(f"\nFinal structured-pruned accuracy: {acc_after:.2f}%")
    print(f"Checkpoint size {file_size_mb(STRUCT_PATH):.2f} MB vs FP32 {file_size_mb(FP32_PATH):.2f} MB")
    print("Saved structured pruned model ->", STRUCT_PATH)


if __name__ == "__main__":
    main()
