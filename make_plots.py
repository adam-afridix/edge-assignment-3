# make_plots.py
# Makes the figures for the report from the CSV files in results/
# usage:  python make_plots.py

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULT_DIR

FIG_DIR = os.path.join(RESULT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# every model keeps the same color in every figure
COLORS = {
    "fp32": "#2a78d6", "ptq_int8": "#eb6834", "qat_int8": "#1baf7a",
    "unstructured": "#eda100", "structured": "#e87ba4",
    "unstructured_pruned": "#eda100", "structured_pruned": "#e87ba4",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def bars(ax, names, values, title, unit, fmt="{:.2f}"):
    x = range(len(names))
    ax.bar(x, values, width=0.6, color=[COLORS[n] for n in names], edgecolor="white", linewidth=2)
    for i, v in enumerate(values):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels([n.replace("_pruned", "\npruned").replace("_int8", "\nint8") for n in names])
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylabel(unit)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(values) * 1.18)


def summary_figure(csv_name, out_name, title, size_col="size_MB", size_label="checkpoint size"):
    path = os.path.join(RESULT_DIR, csv_name)
    if not os.path.exists(path):
        print("skip", csv_name)
        return
    df = pd.read_csv(path)
    names = df["model"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(12 if len(names) <= 3 else 15.5, 3.8))
    bars(axes[0], names, df["test_accuracy_%"], "Test accuracy", "%")
    lo = df["test_accuracy_%"].min()
    axes[0].set_ylim(max(0, lo - 5), 100)
    bars(axes[1], names, df[size_col], size_label[0].upper() + size_label[1:], "MB")
    bars(axes[2], names, df["latency_median_ms_bs1"], "CPU latency (batch 1, median)", "ms / image")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, out_name), dpi=160)
    plt.close(fig)
    print("saved", out_name)


def unstructured_figure():
    path = os.path.join(RESULT_DIR, "unstructured_iterations.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(df["actual_sparsity_%"], df["acc_after_prune_%"], "-o", color="#eb6834", lw=2, ms=6,
            label="right after pruning")
    ax.plot(df["actual_sparsity_%"], df["acc_after_finetune_%"], "-o", color="#2a78d6", lw=2, ms=6,
            label="after fine-tuning")
    for _, r in df.iterrows():
        ax.annotate(f"{r['acc_after_finetune_%']:.1f}", (r["actual_sparsity_%"], r["acc_after_finetune_%"]),
                    textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=INK)
    ax.set_xlabel("weight sparsity (%)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Iterative magnitude pruning", loc="left")
    ax.grid(True, color=GRID, lw=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q2_unstructured_iterations.png"), dpi=160)
    plt.close(fig)
    print("saved q2_unstructured_iterations.png")


def calibration_figure():
    path = os.path.join(RESULT_DIR, "ptq_calibration_size.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(df["calibration_images"], df["ptq_int8_test_acc"], "-o", color="#eb6834", lw=2, ms=6)
    for _, r in df.iterrows():
        ax.annotate(f"{r['ptq_int8_test_acc']:.2f}", (r["calibration_images"], r["ptq_int8_test_acc"]),
                    textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(df["calibration_images"])
    ax.set_xticklabels(df["calibration_images"])
    ax.set_xlabel("number of calibration images (log scale)")
    ax.set_ylabel("PTQ INT8 test accuracy (%)")
    ax.set_title("Effect of calibration set size on PTQ", loc="left")
    ax.grid(True, color=GRID, lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "q1_calibration_size.png"), dpi=160)
    plt.close(fig)
    print("saved q1_calibration_size.png")


if __name__ == "__main__":
    summary_figure("q1_benchmark.csv", "q1_quantization_benchmark.png", "Question 1 - FP32 vs PTQ vs QAT (PyTorch, CPU)")
    summary_figure("q2_benchmark.csv", "q2_pruning_benchmark.png", "Question 2 - original vs pruned models (PyTorch, CPU)",
                   size_col="zipped_size_MB", size_label="zipped checkpoint size")
    summary_figure("onnx_benchmark.csv", "onnx_benchmark.png", "ONNX Runtime on CPU - all five models",
                   size_col="onnx_size_MB", size_label="ONNX file size")
    unstructured_figure()
    calibration_figure()
