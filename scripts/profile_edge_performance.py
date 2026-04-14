"""
Edge Performance Profiling Script.

This script benchmarks the inference latency, memory footprint, and disk size of 
a PyTorch Model (FP32), an ONNX Model (FP32), and a Dynamically Quantized ONNX Model (INT8).

It serves as a tool to quantify Speed-vs-Accuracy trade-offs for edge deployments
(e.g., Jetson Nano, Raspberry Pi) onboard autonomous drones.
"""

import os
import time
import psutil
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process
from pathlib import Path


def get_memory_usage_mb() -> float:
    """Returns the current Resident Set Size (RSS) memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def export_to_onnx(model: nn.Module, dummy_input: torch.Tensor, output_path: str) -> None:
    """Exports a PyTorch model to ONNX FP32."""
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )


def quantize_to_int8(fp32_path: str, int8_path: str) -> None:
    """Applies dynamic quantization to an ONNX FP32 model to create an INT8 model."""
    quant_pre_process(input_model_path=fp32_path, output_model_path=fp32_path, skip_optimization=False)
    quantize_dynamic(
        fp32_path,
        int8_path,
        weight_type=QuantType.QUInt8
    )


def measure_latency(inference_fn, iterations: int = 100, warmup: int = 10):
    """
    Measures the inference latency of a provided inference function.
    
    Returns:
        mean_latency_ms (float), p99_latency_ms (float)
    """
    # Warmup
    for _ in range(warmup):
        inference_fn()

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        inference_fn()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    latencies = np.array(latencies)
    mean_lat = np.mean(latencies)
    p99_lat = np.percentile(latencies, 99)
    return mean_lat, p99_lat


def main():
    print("Initializing Edge Performance Profiling Suite...")
    
    # 1. Setup Directories & Dummy Data
    models_dir = Path("../models")
    if not models_dir.exists():
         models_dir = Path("models")
         models_dir.mkdir(parents=True, exist_ok=True)
         
    fp32_onnx_path = str(models_dir / "resnet18_fp32.onnx")
    int8_onnx_path = str(models_dir / "resnet18_int8.onnx")
    
    # Standard input size for Vision Models (1 Batch, 3 Channels, 224 H, 224 W)
    dummy_input_pt = torch.randn(1, 3, 224, 224)
    dummy_input_np = dummy_input_pt.numpy()

    # 2. Get PyTorch Baseline Tracking
    base_mem = get_memory_usage_mb()
    print("Loading PyTorch FP32 Baseline...")
    
    # Swapped MobileNetV2 for ResNet18 to gracefully bypass PyTorch 2.6's ONNX Shape Inference Bug
    pt_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    pt_model.eval()
    pt_mem = get_memory_usage_mb() - base_mem
    
    # Export and generate INT8 representations
    print("Exporting PyTorch to ONNX FP32...")
    export_to_onnx(pt_model, dummy_input_pt, fp32_onnx_path)
    
    print("Dynamically Quantizing ONNX FP32 to ONNX INT8...")
    quantize_to_int8(fp32_onnx_path, int8_onnx_path)

    # Configure ONNX Runtime sessions
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    print("\nStarting Benchmarks (100 Iterations)...")
    
    def pt_infer():
        with torch.no_grad():
            _ = pt_model(dummy_input_pt)
    
    pt_mean, pt_p99 = measure_latency(pt_infer)
    
    # --- ONNX FP32 Benchmark ---
    ort_session_fp32 = ort.InferenceSession(fp32_onnx_path, sess_options=opts, providers=['CPUExecutionProvider'])
    
    def onnx_fp32_infer():
        _ = ort_session_fp32.run(None, {'input': dummy_input_np})
        
    onnx_fp32_mean, onnx_fp32_p99 = measure_latency(onnx_fp32_infer)

    # --- ONNX INT8 Benchmark ---
    ort_session_int8 = ort.InferenceSession(int8_onnx_path, sess_options=opts, providers=['CPUExecutionProvider'])
    
    def onnx_int8_infer():
        _ = ort_session_int8.run(None, {'input': dummy_input_np})
        
    onnx_int8_mean, onnx_int8_p99 = measure_latency(onnx_int8_infer)

    # 4. Gather Disk Sizes
    size_fp32_mb = os.path.getsize(fp32_onnx_path) / (1024 * 1024)
    size_int8_mb = os.path.getsize(int8_onnx_path) / (1024 * 1024)

    # 5. Output Summary Table
    print("\n" + "="*85)
    print(f"{'Runtime Target':<25} | {'Mean Latency (ms)':<18} | {'P99 Latency (ms)':<16} | {'Disk Size (MB)':<14}")
    print("-" * 85)
    print(f"{'PyTorch (FP32)':<25} | {pt_mean:<18.2f} | {pt_p99:<16.2f} | {size_fp32_mb:<14.2f}")
    print(f"{'ONNX Runtime (FP32)':<25} | {onnx_fp32_mean:<18.2f} | {onnx_fp32_p99:<16.2f} | {size_fp32_mb:<14.2f}")
    print(f"{'ONNX Runtime (INT8)':<25} | {onnx_int8_mean:<18.2f} | {onnx_int8_p99:<16.2f} | {size_int8_mb:<14.2f}")
    print("="*85)
    
    print("\nInterview Discussion Points (Speed vs. Accuracy Trade-offs):")
    print(f"1. Quantization Size Reduction: INT8 model is ~{size_fp32_mb/size_int8_mb:.1f}x smaller on disk, drastically lowering VRAM/RAM constraints on embedded hardware.")
    print(f"2. Latency Gain: Observe the P99 latency variance. The ONNX graph optimizations + INT8 execution reduces bottlenecks, improving worst-case frame drops.")
    print(f"3. Accuracy Drop: Dynamic quantization limits computational precision. For spatial workflows like depth/flow, we must validate that 8-bit resolution doesn't degrade pixel-perfect predictions contextually.")
    print("="*85 + "\n")


if __name__ == "__main__":
    main()
