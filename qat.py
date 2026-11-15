# qat.py
# Quantization-Aware Training (QAT): FP32 -> fake-quant fine-tuning -> INT8
#
# Steps:
#   1. start from the trained FP32 checkpoint
#   2. fuse layers (QAT version keeps BN so it can still train)
#   3. set the QAT qconfig and call prepare_qat -> this inserts FakeQuantize
#      modules on weights and activations (they round values to INT8 levels
#      in the forward pass but gradients still flow, straight-through estimator)
#   4. fine-tune a few epochs with a small learning rate
#      - BN statistics are frozen after a few epochs
#      - observers are frozen after a few epochs (scale/zero-point stop moving)
#   5. convert the fine-tuned model to real INT8 on CPU and save it
#
# usage:  python qat.py --epochs 5

import argparse
import copy
import time
import torch
import torch.ao.quantization as tq
import torch.ao.nn.intrinsic.qat as nniqat

from common import (ResNet18, load_fp32, fuse_model, get_loaders, get_test_loader, train_one_epoch,
                    evaluate, get_device, set_seed, file_size_mb,
                    QAT_PATH, QAT_FAKEQ_PATH, QBACKEND)


def build_qat_prepared(fp32_model, backend=QBACKEND):
    """FP32 model -> fused -> FakeQuantize modules inserted (still a float model)."""
    torch.backends.quantized.engine = backend
    model = copy.deepcopy(fp32_model).cpu()
    model.train()                       # fuse_modules_qat needs train mode
    fuse_model(model, qat=True)
    # version=0 -> normal FakeQuantize modules. The default (version=1) uses a fused
    # op (fused_moving_avg_obs_fake_quant) which cannot be exported to ONNX.
    model.qconfig = tq.get_default_qat_qconfig(backend, version=0)
    tq.prepare_qat(model, inplace=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--freeze_bn", type=int, default=3, help="freeze BN stats after this epoch")
    parser.add_argument("--freeze_obs", type=int, default=4, help="freeze observers after this epoch")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    print("Device:", device, "| backend:", QBACKEND)

    train_loader, test_loader = get_loaders(128, device)
    fp32 = load_fp32()
    print(f"FP32 starting accuracy: {evaluate(fp32.to(device), test_loader, device):.2f}%")
    fp32.cpu()

    model = build_qat_prepared(fp32).to(device)
    print("\nFake-quant modules inserted, first block now looks like:")
    print(model.layer1[0].conv1)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        if ep > args.freeze_bn:
            model.apply(nniqat.freeze_bn_stats)
        if ep > args.freeze_obs:
            model.apply(tq.disable_observer)
        # no mixed precision here, fake quant should run in normal fp32
        loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, scheduler, None)
        fq_acc = evaluate(model, test_loader, device)
        print(f"QAT epoch {ep}/{args.epochs} | loss {loss:.4f} | train acc {tr_acc:.2f}% | "
              f"fake-quant test acc {fq_acc:.2f}% | {time.time() - t0:.1f}s")

    # keep the fake-quant weights too, the ONNX export uses them
    model.cpu()
    torch.save(model.state_dict(), QAT_FAKEQ_PATH)

    # convert to real INT8 (must be on CPU + eval)
    model.eval()
    int8_model = tq.convert(model, inplace=False)

    scripted = torch.jit.script(int8_model)
    torch.jit.save(scripted, QAT_PATH)
    print("\nSaved INT8 QAT model ->", QAT_PATH)

    # load back and test
    loaded = torch.jit.load(QAT_PATH)
    acc = evaluate(loaded, get_test_loader("cpu"))
    print(f"QAT INT8 test acc: {acc:.2f}%   size {file_size_mb(QAT_PATH):.2f} MB")


if __name__ == "__main__":
    main()
