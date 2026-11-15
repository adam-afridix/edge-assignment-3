# export_onnx.py
# Exports the models to ONNX so they can run with ONNX Runtime on any CPU
# (no PyTorch needed at deployment time).
#
#   fp32.onnx                 <- baseline FP32 model
#   unstructured_pruned.onnx  <- magnitude pruned model
#   structured_pruned.onnx    <- channel pruned (smaller) model
#   qat_int8.onnx             <- QAT model. The fake-quant modules are exported as
#                                QuantizeLinear/DequantizeLinear (QDQ) nodes, ONNX Runtime
#                                then runs the conv layers with INT8 kernels.
#   ptq_int8.onnx             <- PTQ done with ONNX Runtime's static quantizer on fp32.onnx,
#                                calibrated with the same 2000 training images as ptq.py.
#                                (PyTorch's converted INT8 modules can not be exported
#                                directly, so this is the normal way to get a PTQ ONNX model.)
#
# Every export is checked: onnx.checker + the ONNX output is compared with PyTorch.
#
# usage:  python export_onnx.py

import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_weights
import torch.ao.nn.intrinsic.qat as nniqat
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (quantize_static, CalibrationDataReader, QuantFormat,
                                      QuantType, CalibrationMethod)
from onnxruntime.quantization.shape_inference import quant_pre_process
import torch.ao.quantization as tq

from common import (ResNet18, load_fp32, load_qat, load_unstructured, load_structured, get_calib_loader,
                    get_test_set, file_size_mb, ONNX_DIR, QAT_FAKEQ_PATH)
from qat import build_qat_prepared

warnings.filterwarnings("ignore")
OPSET = 13


def export(model, path):
    model.eval()
    dummy = torch.randn(1, 3, 32, 32)
    torch.onnx.export(model, dummy, path, opset_version=OPSET,
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      do_constant_folding=True)
    onnx.checker.check_model(onnx.load(path))


def compare(model, path, n=64):
    """run the same test images through PyTorch and ONNX Runtime"""
    ts = get_test_set()
    x = torch.stack([ts[i][0] for i in range(n)])
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": x.numpy()})[0]
    diff = np.abs(ref - out).max()
    agree = (ref.argmax(1) == out.argmax(1)).mean() * 100
    return diff, agree


def fold_weight_quantize(path):
    """The fake-quant export stores weights as FP32 followed by QuantizeLinear.
    Here we do that QuantizeLinear once, offline, and store the INT8 weights
    directly in the file (QuantizeLinear(W) -> int8 initializer -> DequantizeLinear).
    Same result, but the file is ~4x smaller. This is the same layout ONNX Runtime's
    own quantizer produces."""
    from onnx import numpy_helper
    m = onnx.load(path)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    producer = {o: n for n in g.node for o in n.output}

    def value(name):
        """constant value of a tensor (initializer, Constant node or Identity of one)"""
        if name in inits:
            return numpy_helper.to_array(inits[name])
        n = producer.get(name)
        if n is None:
            return None
        if n.op_type == "Constant":
            return numpy_helper.to_array(n.attribute[0].t)
        if n.op_type == "Identity":
            return value(n.input[0])
        return None

    remove, folded = [], 0
    for n in g.node:
        if n.op_type != "QuantizeLinear" or n.input[0] not in inits:
            continue
        w = numpy_helper.to_array(inits[n.input[0]]).astype(np.float32)
        scale, zp = value(n.input[1]), value(n.input[2])
        if scale is None or zp is None:
            continue
        axis = next((a.i for a in n.attribute if a.name == "axis"), 1)
        if scale.ndim == 1 and scale.size > 1:          # per-channel -> reshape for broadcasting
            shape = [1] * w.ndim
            shape[axis] = -1
            scale_b, zp_b = scale.reshape(shape), zp.reshape(shape).astype(np.float32)
        else:
            scale_b, zp_b = scale, zp.astype(np.float32)
        info = np.iinfo(zp.dtype)
        q = np.clip(np.round(w / scale_b) + zp_b, info.min, info.max).astype(zp.dtype)
        g.initializer.append(numpy_helper.from_array(q, name=n.output[0]))
        remove.append(n)
        folded += 1
    for n in remove:
        g.node.remove(n)
    # drop float weights / Identity nodes that nothing uses anymore
    changed = True
    while changed:
        used = {i for n in g.node for i in n.input} | {o.name for o in g.output}
        dead = [n for n in g.node if n.op_type in ("Identity", "Constant") and n.output[0] not in used]
        for n in dead:
            g.node.remove(n)
        changed = len(dead) > 0
    used = {i for n in g.node for i in n.input}
    for init in list(g.initializer):
        if init.name not in used:
            g.initializer.remove(init)
    onnx.save(m, path)
    return folded


def requantize_relu(path):
    """In every block the last ReLU comes after the quantized add:  Q -> DQ -> Relu -> next conv.
    The next conv then gets a float input and ONNX Runtime can not use the INT8 conv kernel.
    ReLU(x) = max(x, 0) and 0 is on the quantization grid, so the ReLU output can be quantized
    again with the SAME scale / zero-point without any error. We add that Q -> DQ pair after
    each such ReLU, then the next conv becomes DQ -> Conv -> Q and gets fused (QLinearConv)."""
    m = onnx.load(path)
    g = m.graph
    producer = {o: n for n in g.node for o in n.output}
    new_nodes, count = [], 0
    for n in g.node:
        new_nodes.append(n)
        if n.op_type != "Relu" or producer.get(n.input[0]) is None:
            continue
        dq = producer[n.input[0]]
        if dq.op_type != "DequantizeLinear":
            continue
        old_out = n.output[0]
        relu_out = old_out + "_float"
        n.output[0] = relu_out
        q_out = old_out + "_q"
        new_nodes.append(onnx.helper.make_node("QuantizeLinear", [relu_out, dq.input[1], dq.input[2]], [q_out]))
        new_nodes.append(onnx.helper.make_node("DequantizeLinear", [q_out, dq.input[1], dq.input[2]], [old_out]))
        count += 1
    del g.node[:]
    g.node.extend(new_nodes)
    onnx.save(m, path)
    return count


class FoldedQATConv(nn.Module):
    """QAT ConvBn(ReLU) with the BatchNorm folded into the conv weight and bias.
    PyTorch's QAT ConvBn module exports as Conv -> Div -> BatchNormalization, which is
    slow and keeps float weights. convert() folds BN in exactly this way, so the
    ONNX model now matches the INT8 PyTorch QAT model. The learned fake-quant
    modules (scale / zero-point from QAT) are kept as they are."""

    def __init__(self, m):
        super().__init__()
        bn = m.bn
        w, b = fuse_conv_bn_weights(m.weight, m.bias, bn.running_mean, bn.running_var,
                                    bn.eps, bn.weight, bn.bias)
        self.weight = nn.Parameter(w.detach())
        self.bias = nn.Parameter(b.detach())
        self.stride, self.padding, self.dilation, self.groups = m.stride, m.padding, m.dilation, m.groups
        self.weight_fake_quant = m.weight_fake_quant
        self.out_fake_quant = m.activation_post_process
        self.relu = isinstance(m, nniqat.ConvBnReLU2d)

    def forward(self, x):
        y = F.conv2d(x, self.weight_fake_quant(self.weight), self.bias,
                     self.stride, self.padding, self.dilation, self.groups)
        if self.relu:
            y = F.relu(y)
        return self.out_fake_quant(y)


def full_uint8_range(model):
    """x86 qconfig uses reduce_range for activations (quant range 0..127). That only
    changes how the scale is computed; the real INT8 PyTorch model (after convert)
    still stores activations as quint8 0..255 with the same scale and zero-point.
    If we export with 0..127, torch adds a Clip node after every QuantizeLinear,
    which stops ONNX Runtime from fusing the INT8 kernels. Setting quant_max to 255
    (scale and zero-point stay the same) gives clean Q/DQ pairs and matches the
    converted PyTorch model."""
    n = 0
    for m in model.modules():
        if isinstance(m, tq.FakeQuantize) and m.quant_min == 0 and m.quant_max == 127:
            m.quant_max = 255
            m.activation_post_process.quant_max = 255   # forward() reads the range from here
            n += 1
    return n


def fold_bn_qat(model):
    """replace every QAT ConvBn / ConvBnReLU module with FoldedQATConv"""
    for name, m in list(model.named_modules()):
        if isinstance(m, nniqat.ConvBn2d):            # ConvBnReLU2d is a subclass of ConvBn2d
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, FoldedQATConv(m))
    return model


class CalibReader(CalibrationDataReader):
    """gives ONNX Runtime the calibration batches one by one"""

    def __init__(self, n_images=2000):
        self.batches = iter([x.numpy() for x, _ in get_calib_loader(n_images, batch_size=50)])

    def get_next(self):
        b = next(self.batches, None)
        return None if b is None else {"input": b}


def main():
    results = []

    # ---------- float models
    for name, loader in [("fp32", load_fp32), ("unstructured_pruned", load_unstructured),
                         ("structured_pruned", load_structured)]:
        path = os.path.join(ONNX_DIR, f"{name}.onnx")
        model = loader()
        export(model, path)
        diff, agree = compare(model, path)
        results.append((name, path, diff, agree))

    # ---------- QAT (fake quant -> QDQ)
    path = os.path.join(ONNX_DIR, "qat_int8.onnx")
    qat_model = build_qat_prepared(ResNet18())
    qat_model.load_state_dict(torch.load(QAT_FAKEQ_PATH, map_location="cpu"))
    qat_model.apply(tq.disable_observer)
    qat_model.eval()
    fold_bn_qat(qat_model)
    full_uint8_range(qat_model)
    export(qat_model, path)
    n_folded = fold_weight_quantize(path)
    n_relu = requantize_relu(path)
    onnx.checker.check_model(onnx.load(path))
    print(f"QAT ONNX: {n_folded} weight tensors stored as INT8, {n_relu} ReLU outputs re-quantized")
    diff, agree = compare(load_qat(), path)     # compared with the real INT8 PyTorch QAT model
    results.append(("qat_int8", path, diff, agree))

    # ---------- PTQ with ONNX Runtime
    fp32_path = os.path.join(ONNX_DIR, "fp32.onnx")
    prep_path = os.path.join(ONNX_DIR, "fp32_prep.tmp.onnx")
    ptq_path = os.path.join(ONNX_DIR, "ptq_int8.onnx")
    quant_pre_process(fp32_path, prep_path)          # shape inference + graph clean-up
    quantize_static(prep_path, ptq_path, CalibReader(2000),
                    quant_format=QuantFormat.QDQ,
                    per_channel=True,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax)
    os.remove(prep_path)
    onnx.checker.check_model(onnx.load(ptq_path))
    diff, agree = compare(load_fp32(), ptq_path)
    results.append(("ptq_int8", ptq_path, diff, agree))

    print(f"\n{'model':<22}{'size MB':>9}{'max |diff| vs torch':>22}{'same top-1 %':>14}")
    for name, p, d, a in results:
        print(f"{name:<22}{file_size_mb(p):>9.2f}{d:>22.5f}{a:>14.1f}")
    print("(qat_int8 is compared with the INT8 PyTorch QAT model, ptq_int8 with the FP32 PyTorch model)")
    print("\nONNX models saved in", ONNX_DIR)


if __name__ == "__main__":
    main()
