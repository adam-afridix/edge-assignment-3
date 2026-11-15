#!/bin/bash
# run_all.sh - same pipeline as run_all.ps1 for Linux / macOS / Git Bash
#   bash run_all.sh
set -e
mkdir -p logs

step () {
    echo ""
    echo "==================== $1 ===================="
    python -u $3 2>&1 | tee logs/$2
}

step "0. Prepare data"          00_prepare_data.log       "prepare_data.py"
step "1. Train FP32 baseline"   01_train_baseline.log     "train_baseline.py --epochs 30"
step "2. PTQ calibration study" 02_calibration.log        "calibration.py"
step "3. PTQ (INT8)"            03_ptq.log                "ptq.py --calib 2000"
step "4. QAT (INT8)"            04_qat.log                "qat.py --epochs 5"
step "5. Benchmark Q1"          05_benchmark_q1.log       "benchmark.py --group quant"
step "6. Unstructured pruning"  06_prune_unstructured.log "prune_unstructured.py"
step "7. Structured pruning"    07_prune_structured.log   "prune_structured.py --ratio 0.5 --ft_epochs 10"
step "8. Benchmark Q2"          08_benchmark_q2.log       "benchmark.py --group prune"
step "9. Inference (PyTorch)"   09_inference.log          "inference.py --model all --samples 6"
step "10. Export ONNX"          10_export_onnx.log        "export_onnx.py"
step "11. ONNX benchmark"       11_onnx_benchmark.log     "onnx_inference.py --benchmark"
step "12. ONNX inference"       12_onnx_inference.log     "onnx_inference.py --model all --samples 6"
step "13. Plots"                13_plots.log              "make_plots.py"

echo "All done. See results/ and logs/"
