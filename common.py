# common.py
# Shared code for Assignment 3 (Edge Computing & Intelligent Systems)
# Adam Afridi - 23i-0055 - AI-B
#
# This file has the things every other script needs:
#   - CIFAR-10 data loaders
#   - ResNet-18 (CIFAR version) that can be quantized and channel-pruned
#   - train / evaluate helpers
#   - functions to load each of the five saved models
#   - small helpers for timing and file size

import os
import time
import random
import zipfile
import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.ao.quantization import QuantStub, DeQuantStub

# folders
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
ONNX_DIR = os.path.join(ROOT, "onnx_models")
RESULT_DIR = os.path.join(ROOT, "results")
for d in [MODEL_DIR, ONNX_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# file names of the saved models
FP32_PATH = os.path.join(MODEL_DIR, "fp32_resnet18.pth")
PTQ_PATH = os.path.join(MODEL_DIR, "ptq_int8_scripted.pt")
QAT_PATH = os.path.join(MODEL_DIR, "qat_int8_scripted.pt")
QAT_FAKEQ_PATH = os.path.join(MODEL_DIR, "qat_fakequant_state.pth")   # used for ONNX export
UNSTRUCT_PATH = os.path.join(MODEL_DIR, "unstructured_pruned.pth")
STRUCT_PATH = os.path.join(MODEL_DIR, "structured_pruned.pth")

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# CIFAR-10 mean and std (per channel)
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

# quantization backend (x86 is the default one for Intel/AMD CPUs)
QBACKEND = "x86" if "x86" in torch.backends.quantized.supported_engines else "fbgemm"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
test_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


class CIFAR10(torch.utils.data.Dataset):
    """CIFAR-10 read from the .npz files made by prepare_data.py"""

    def __init__(self, train=True, transform=None):
        path = os.path.join(DATA_DIR, f"cifar10_{'train' if train else 'test'}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found, run  python prepare_data.py  first")
        d = np.load(path)
        self.data = d["x"]                     # (N, 32, 32, 3) uint8
        self.targets = d["y"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img = Image.fromarray(self.data[i])
        if self.transform:
            img = self.transform(img)
        return img, self.targets[i]


class TensorLoader:
    """Faster loader used for training.
    The normal DataLoader was very slow on Windows (~95 s/epoch, the GPU was waiting
    for PIL augmentations). Here the whole dataset is one tensor on the device and the
    augmentation (random crop with 4px padding + horizontal flip) is done with tensor
    ops on the full batch. Gives ~same augmentation as train_tf."""

    def __init__(self, train, batch_size, device, augment, shuffle):
        d = np.load(os.path.join(DATA_DIR, f"cifar10_{'train' if train else 'test'}.npz"))
        x = torch.tensor(d["x"]).permute(0, 3, 1, 2).float() / 255.0
        mean = torch.tensor(MEAN).view(1, 3, 1, 1)
        std = torch.tensor(STD).view(1, 3, 1, 1)
        self.x = ((x - mean) / std).to(device)
        self.y = torch.tensor(d["y"]).long().to(device)
        self.bs, self.augment, self.shuffle = batch_size, augment, shuffle

    def __len__(self):
        return (len(self.y) + self.bs - 1) // self.bs

    def _augment(self, xb):
        n, dev = xb.size(0), xb.device
        xb = F.pad(xb, (4, 4, 4, 4))                       # 40x40
        dy = torch.randint(0, 9, (n,), device=dev)
        dx = torch.randint(0, 9, (n,), device=dev)
        ar = torch.arange(32, device=dev)
        rows = (dy[:, None] + ar)[:, None, :, None]         # n,1,32,1
        cols = (dx[:, None] + ar)[:, None, None, :]         # n,1,1,32
        xb = xb[torch.arange(n, device=dev)[:, None, None, None],
                torch.arange(3, device=dev)[None, :, None, None], rows, cols]
        flip = torch.rand(n, device=dev) < 0.5
        xb[flip] = xb[flip].flip(3)
        return xb

    def __iter__(self):
        n = len(self.y)
        idx = torch.randperm(n, device=self.y.device) if self.shuffle else torch.arange(n, device=self.y.device)
        for i in range(0, n, self.bs):
            b = idx[i:i + self.bs]
            xb = self.x[b]
            if self.augment:
                xb = self._augment(xb)
            yield xb, self.y[b]


def get_loaders(batch_size=128, device="cpu"):
    """train loader (with augmentation) and test loader, data already on `device`"""
    train_loader = TensorLoader(True, batch_size, device, augment=True, shuffle=True)
    test_loader = TensorLoader(False, 256, device, augment=False, shuffle=False)
    return train_loader, test_loader


def get_test_loader(device="cpu"):
    return TensorLoader(False, 256, device, augment=False, shuffle=False)


def get_calib_loader(num_images=2000, batch_size=100, seed=0):
    """Random subset of the TRAINING set (no augmentation) used for calibration."""
    train_set = CIFAR10(train=True, transform=test_tf)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(train_set), generator=g)[:num_images].tolist()
    subset = torch.utils.data.Subset(train_set, idx)
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)


def get_test_set():
    return CIFAR10(train=False, transform=test_tf)


# ----------------------------------------------------------------------
# Model: ResNet-18 for 32x32 images
# ----------------------------------------------------------------------
# Changes from the normal ImageNet ResNet-18:
#   * first conv is 3x3 stride 1 and there is no maxpool (images are only 32x32)
#   * every ReLU is its own module and the residual add uses FloatFunctional,
#     this is needed for eager mode quantization (fusion + quantized add)
#   * "mid" = number of channels between conv1 and conv2 of each block.
#     Structured pruning makes this smaller, the block input/output stays same.

class BasicBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=False)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Sequential()   # identity

        # quantization friendly add
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.skip_add.add(out, self.shortcut(x))
        out = self.relu2(out)
        return out


# default channel config of ResNet-18 (2 blocks per stage)
STAGE_CH = [64, 128, 256, 512]
DEFAULT_MIDS = [64, 64, 128, 128, 256, 256, 512, 512]


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, mids=None):
        super().__init__()
        if mids is None:
            mids = list(DEFAULT_MIDS)
        self.mids = list(mids)

        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)

        blocks = []
        in_ch = 64
        k = 0
        for stage, out_ch in enumerate(STAGE_CH):
            for b in range(2):
                stride = 2 if (stage > 0 and b == 0) else 1
                blocks.append(BasicBlock(in_ch, mids[k], out_ch, stride))
                in_ch = out_ch
                k += 1
        self.layer1 = nn.Sequential(blocks[0], blocks[1])
        self.layer2 = nn.Sequential(blocks[2], blocks[3])
        self.layer3 = nn.Sequential(blocks[4], blocks[5])
        self.layer4 = nn.Sequential(blocks[6], blocks[7])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.dequant = DeQuantStub()

    def blocks(self):
        """list of the 8 BasicBlocks in order"""
        return list(self.layer1) + list(self.layer2) + list(self.layer3) + list(self.layer4)

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)
        return x


def fuse_model(model, qat=False):
    """Fuse conv+bn(+relu) groups. Model must be in eval mode for PTQ and train mode for QAT."""
    fuse = torch.ao.quantization.fuse_modules_qat if qat else torch.ao.quantization.fuse_modules
    fuse(model, [["conv1", "bn1", "relu"]], inplace=True)
    for blk in model.blocks():
        fuse(blk, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
        if len(blk.shortcut) == 2:
            fuse(blk.shortcut, [["0", "1"]], inplace=True)
    return model


# ----------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, scheduler=None, scaler=None):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(x)
                loss = F.cross_entropy(out, y, label_smoothing=0.1)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = F.cross_entropy(out, y, label_smoothing=0.1)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, 100.0 * correct / total


def fine_tune(model, train_loader, test_loader, device, epochs, lr, tag="FT"):
    """Short fine-tuning with SGD + cosine lr. Used after pruning."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    acc = None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, scheduler, scaler)
        acc = evaluate(model, test_loader, device)
        print(f"  [{tag}] epoch {ep}/{epochs} | loss {loss:.4f} | train {tr_acc:.2f}% | "
              f"test {acc:.2f}% | {time.time() - t0:.1f}s")
    return acc


@torch.no_grad()
def evaluate(model, loader, device="cpu", return_preds=False):
    model.eval()
    correct, total = 0, 0
    preds = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = model(x).argmax(1)
        correct += (p == y).sum().item()
        total += y.size(0)
        if return_preds:
            preds.append(p.cpu())
    acc = 100.0 * correct / total
    if return_preds:
        return acc, torch.cat(preds).numpy()
    return acc


# ----------------------------------------------------------------------
# Loading the five models (everything is loaded on CPU)
# ----------------------------------------------------------------------
def load_fp32():
    model = ResNet18()
    model.load_state_dict(torch.load(FP32_PATH, map_location="cpu"))
    return model.eval()


def load_ptq():
    torch.backends.quantized.engine = QBACKEND
    return torch.jit.load(PTQ_PATH, map_location="cpu").eval()


def load_qat():
    torch.backends.quantized.engine = QBACKEND
    return torch.jit.load(QAT_PATH, map_location="cpu").eval()


def load_unstructured():
    model = ResNet18()
    model.load_state_dict(torch.load(UNSTRUCT_PATH, map_location="cpu"))
    return model.eval()


def load_structured():
    # checkpoint keeps the new (smaller) channel config so we can rebuild the network
    ckpt = torch.load(STRUCT_PATH, map_location="cpu")
    model = ResNet18(mids=ckpt["mids"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


MODEL_FILES = {
    "fp32": FP32_PATH,
    "ptq_int8": PTQ_PATH,
    "qat_int8": QAT_PATH,
    "unstructured": UNSTRUCT_PATH,
    "structured": STRUCT_PATH,
}
LOADERS = {
    "fp32": load_fp32,
    "ptq_int8": load_ptq,
    "qat_int8": load_qat,
    "unstructured": load_unstructured,
    "structured": load_structured,
}


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def zipped_size_mb(path):
    """Size after zip compression. Zero weights compress very well, so this
    shows the real storage gain of unstructured pruning."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(path, arcname=os.path.basename(path))
    return buf.getbuffer().nbytes / (1024 * 1024)


def count_params(model):
    total, nonzero = 0, 0
    for p in model.parameters():
        total += p.numel()
        nonzero += (p != 0).sum().item()
    return total, nonzero


@torch.no_grad()
def measure_latency(model, n_runs=300, warmup=30, batch_size=1):
    """CPU latency (ms) for one forward pass -> (median, p95).
    Median is used because a laptop has other programs running and a few
    slow runs would pull the mean up."""
    model.eval()
    x = torch.randn(batch_size, 3, 32, 32)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


@torch.no_grad()
def measure_throughput(model, loader):
    """Images per second over the whole test set on CPU."""
    model.eval()
    n = 0
    t0 = time.perf_counter()
    for x, _ in loader:
        model(x)
        n += x.size(0)
    return n / (time.perf_counter() - t0)
