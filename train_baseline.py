# train_baseline.py
# Step 1: train the FP32 ResNet-18 on CIFAR-10 and save the checkpoint.
#
# usage:  python train_baseline.py --epochs 30

import argparse
import time
import torch

from common import (ResNet18, get_loaders, train_one_epoch, evaluate,
                    get_device, set_seed, FP32_PATH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    print("Device:", device)

    train_loader, test_loader = get_loaders(args.batch, device)
    model = ResNet18().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    # one cycle schedule, lr goes up then slowly down
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                    epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))
    # mixed precision only when GPU is there (just to train faster)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, scheduler, scaler)
        te_acc = evaluate(model, test_loader, device)
        print(f"Epoch {ep:2d}/{args.epochs} | loss {loss:.4f} | train acc {tr_acc:.2f}% | "
              f"test acc {te_acc:.2f}% | {time.time() - t0:.1f}s")

    # we keep the model from the last epoch (lr is almost 0 at the end of one-cycle)
    torch.save(model.state_dict(), FP32_PATH)
    print(f"\nFinal test accuracy: {te_acc:.2f}%")
    print("Saved FP32 checkpoint ->", FP32_PATH)


if __name__ == "__main__":
    main()
