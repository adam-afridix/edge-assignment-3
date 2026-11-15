# run_all.ps1
# Runs the full pipeline from start to end (Windows PowerShell).
# Every step also writes its console output into logs\
#
#   powershell -ExecutionPolicy Bypass -File run_all.ps1

New-Item -ItemType Directory -Force logs | Out-Null

function Step($name, $log, $cmd) {
    Write-Host "`n==================== $name ====================" -ForegroundColor Cyan
    # run through cmd so python warnings (stderr) are not turned into PowerShell errors
    cmd /c "$cmd 2>&1" | Tee-Object -FilePath "logs\$log"
    if ($LASTEXITCODE -ne 0) { Write-Host "$name failed" -ForegroundColor Red; exit 1 }
}

Step "0. Prepare data"            "00_prepare_data.log"       "python -u prepare_data.py"
Step "1. Train FP32 baseline"     "01_train_baseline.log"     "python -u train_baseline.py --epochs 30"
# ---- Question 1: quantization
Step "2. PTQ calibration study"   "02_calibration.log"        "python -u calibration.py"
Step "3. PTQ (INT8)"              "03_ptq.log"                "python -u ptq.py --calib 2000"
Step "4. QAT (INT8)"              "04_qat.log"                "python -u qat.py --epochs 5"
Step "5. Benchmark Q1"            "05_benchmark_q1.log"       "python -u benchmark.py --group quant"
# ---- Question 2: pruning
Step "6. Unstructured pruning"    "06_prune_unstructured.log" "python -u prune_unstructured.py"
Step "7. Structured pruning"      "07_prune_structured.log"   "python -u prune_structured.py --ratio 0.5 --ft_epochs 10"
Step "8. Benchmark Q2"            "08_benchmark_q2.log"       "python -u benchmark.py --group prune"
# ---- deployment
Step "9. Inference (PyTorch)"     "09_inference.log"          "python -u inference.py --model all --samples 6"
Step "10. Export ONNX"            "10_export_onnx.log"        "python -u export_onnx.py"
Step "11. ONNX benchmark"         "11_onnx_benchmark.log"     "python -u onnx_inference.py --benchmark"
Step "12. ONNX inference"         "12_onnx_inference.log"     "python -u onnx_inference.py --model all --samples 6"
Step "13. Plots"                  "13_plots.log"              "python -u make_plots.py"

Write-Host "`nAll done. See results\ and logs\" -ForegroundColor Green
