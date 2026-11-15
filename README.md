# Assignment 3 - Model Quantization and Pruning (Edge Computing & Intelligent Systems)

**Name:** Adam Afridi  **Roll No:** 23i-0055  **Section:** AI-B

Model: ResNet-18 (CIFAR version) on CIFAR-10, made smaller and faster with
quantization (Q1) and pruning (Q2), then exported to ONNX for CPU deployment.

---

## 1. Folder structure

```
common.py               shared code: data, ResNet-18 model, train/eval, model loaders, helpers
prepare_data.py         downloads CIFAR-10 once and saves data/cifar10_train.npz, cifar10_test.npz
train_baseline.py       trains the FP32 baseline                      -> models/fp32_resnet18.pth

# Question 1 - quantization
calibration.py          calibration pipeline (+ small study of calibration size)
ptq.py                  post-training static quantization             -> models/ptq_int8_scripted.pt
qat.py                  quantization-aware training                   -> models/qat_int8_scripted.pt

# Question 2 - pruning
prune_unstructured.py   iterative magnitude pruning + fine-tuning     -> models/unstructured_pruned.pth
prune_structured.py     L1 channel pruning, rebuild, fine-tuning      -> models/structured_pruned.pth

# deployment / evaluation
inference.py            PyTorch CPU inference for all five models
benchmark.py            automatic benchmark -> results/*_benchmark.csv and *_predictions.csv
export_onnx.py          exports all five models to onnx_models/
onnx_inference.py       ONNX Runtime CPU inference + ONNX benchmark
make_plots.py           figures for the report -> results/figures/
perf_mode.py            stops Windows power throttling during benchmarks (see notes)
run_all.ps1 / run_all.sh  runs the full pipeline in order

models/        the five PyTorch checkpoints (+ QAT fake-quant weights used for ONNX export)
onnx_models/   the five ONNX models
results/       CSV results and figures
logs/          console output of every step (from my run)
report/        technical report
```

## 2. Installation

Python 3.10+ (I used 3.12.5). A GPU is only used for training; all inference
and benchmarking runs on the CPU.

```bash
# PyTorch with CUDA 12.1 (or use the CPU build from pytorch.org if you have no GPU)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 3. How to run

### Everything at once

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File run_all.ps1
```
```bash
# Linux / macOS / Git Bash
bash run_all.sh
```

### Step by step

| # | command | what it does |
|---|---------|--------------|
| 0 | `python prepare_data.py` | downloads CIFAR-10 (Hugging Face mirror, falls back to the official site) |
| 1 | `python train_baseline.py --epochs 30` | trains the FP32 ResNet-18 |
| 2 | `python calibration.py` | (optional) PTQ accuracy vs number of calibration images |
| 3 | `python ptq.py --calib 2000` | fuse, calibrate on 2000 train images, convert to INT8, save |
| 4 | `python qat.py --epochs 5` | QAT fine-tuning from the FP32 model, convert to INT8, save |
| 5 | `python benchmark.py --group quant` | benchmarks FP32 / PTQ / QAT -> `results/q1_*.csv` |
| 6 | `python prune_unstructured.py` | 5 pruning iterations up to 80% sparsity, fine-tune, make permanent |
| 7 | `python prune_structured.py --ratio 0.5` | removes 50% of the inner channels of every block, rebuilds the net, fine-tunes |
| 8 | `python benchmark.py --group prune` | benchmarks original / unstructured / structured -> `results/q2_*.csv` |
| 9 | `python inference.py --model all --samples 6` | predictions of all five PyTorch models |
| 10 | `python export_onnx.py` | exports the five ONNX models and checks them against PyTorch |
| 11 | `python onnx_inference.py --benchmark` | ONNX Runtime CPU benchmark -> `results/onnx_benchmark.csv` |
| 12 | `python make_plots.py` | figures for the report |

Steps 1, 4, 6 and 7 train on GPU if one is available (about 17, 4, 4 and 4 minutes on an RTX 4050 laptop GPU).
The trained models are already included, so steps 5 and 8-12 can be run straight away
after step 0 (step 0 is needed because the dataset is not included in the zip).

### Inference on your own image

```bash
python inference.py --model qat_int8 --image cat.jpg            # PyTorch
python onnx_inference.py --model structured_pruned --image cat.jpg   # ONNX Runtime
```
Model names (PyTorch): `fp32, ptq_int8, qat_int8, unstructured, structured, all`
Model names (ONNX): `fp32, ptq_int8, qat_int8, unstructured_pruned, structured_pruned, all`

## 4. Output files

| file | content |
|------|---------|
| `results/q1_benchmark.csv` | accuracy, size, latency, throughput, per-class accuracy for FP32, PTQ, QAT |
| `results/q1_predictions.csv` | prediction of every model for every one of the 10,000 test images |
| `results/q2_benchmark.csv` | same for original, unstructured and structured pruned models |
| `results/q2_predictions.csv` | per-image predictions of the pruning models |
| `results/onnx_benchmark.csv` | ONNX Runtime results of all five models |
| `results/ptq_calibration_size.csv` | PTQ accuracy for 100 ... 5000 calibration images |
| `results/unstructured_iterations.csv` | accuracy after every pruning iteration |

## 5. Notes

* Quantized models run on CPU only (PyTorch INT8 kernels are CPU kernels), backend = `x86` (FBGEMM/oneDNN).
* PTQ and QAT models are saved as TorchScript, so they load with `torch.jit.load` without the model code.
* The structured-pruned checkpoint stores the new channel numbers (`mids`) together with the weights, the
  model is rebuilt from that, there are no masks or zero channels left in it.
* For ONNX, the PTQ model is made with ONNX Runtime's static quantizer (same 2000 calibration images), because
  PyTorch's converted INT8 modules can not be exported to ONNX directly. The QAT model is exported from its
  fake-quant version (QuantizeLinear/DequantizeLinear nodes). Before export the BatchNorm layers are folded
  into the convs, and after export the weights are stored as INT8 and the ReLU outputs are re-quantized,
  so ONNX Runtime runs all 20 convolutions with INT8 kernels (QLinearConv). Details in the report.
* QAT uses `get_default_qat_qconfig("x86", version=0)` (normal FakeQuantize). The default version=1 uses a
  fused op that can not be exported to ONNX.
* Windows 11 throttles processes that are not in the foreground (EcoQoS, moves them to efficiency cores).
  On my laptop this made latency jump from ~2 ms to ~60 ms in the middle of a benchmark. `perf_mode.py`
  turns this off for the benchmark process. Latency is reported as the median of 300-500 runs.
* Seeds are fixed (`set_seed(42)`), but GPU training is not 100% bit-exact, so small differences (~0.1-0.3%) are normal.
