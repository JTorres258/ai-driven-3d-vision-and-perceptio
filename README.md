# AI-Driven 3D Vision & Perception for Autonomous Flight

Welcome to the **Drone Perception Workspace**, a cutting-edge repository geared towards the seamless integration of deeply learned visual primitives into autonomous navigation and path-planning workflows.

## Background & Philosophy

This project represents a fundamental transition from classical, heuristic-driven C++ CV pipelines to end-to-end Deep SLAM and multi-modal perception systems. As modern edge hardware capabilities expand, offloading classical geometric formulations to learned deep representations—such as depth and flow estimators—has become critical for scalable, resilient autonomous flight.

## Core Modules

* **Optical Flow (`src/optical_flow.py`)**: Motion estimation powered by RAFT (Recurrent All-Pairs Field Transforms) to robustly track high-velocity dynamics.
* **Monocular Depth (`src/depth_estimation.py`)**: Deep relative depth estimation leveraging the MiDaS architecture to replace or augment active RGB-D sensors in open-air flight.
* **Sensor Fusion (`src/sensor_fusion.py`)**: A Bird's Eye View (BEV) multi-head attention module designed to fuse continuous RGB streams with thermal imaging, yielding robust perception in visually degraded environments.

## Real-Time Inference Application

The script `scripts/run_inference.py` acts as the primary evaluation suite to stream live video data through our multi-modal backbones:
* **Concurrent Perception Loop**: Iterates frame-by-frame, simultaneously extracting motion vector fields (Optical Flow) and relative spatial bounds (Depth).
* **Safety Context Fusion**: Automatically synthesizes collision heuristics using threshold analysis—highlighting pixels exhibiting both high velocity toward the camera and immediate proximity, warning the navigation layer of imminent hazards.
* **Stream Synchronization**: Contains mechanisms to conditionally drop internal hardware grabs when inference delays overrun the strict realtime video framerate cadence.

## Deployment to Edge

Our Deep CV modules must run with minimal latency onboard compute-constrained drones (e.g., Jetson platforms). The script `scripts/export_to_edge.py` facilitates exporting our PyTorch perception backbones directly into optimized ONNX formats with dynamic tensor resolutions, priming them for inference accelerators like TensorRT.

### Edge Performance Profiling

To reliably quantify the speed vs. accuracy trade-offs before hardware deployment, we benchmarked our standard CV backbones targeting a local emulator CPU environment using `scripts/profile_edge_performance.py`:

| Runtime Target            | Mean Latency (ms)  | P99 Latency (ms) | Disk Size (MB) |
|---------------------------|--------------------|------------------|----------------|
| PyTorch (FP32)            | 17.44              | 20.23            | 44.66          |
| ONNX Runtime (FP32)       | 6.60               | 7.29             | 44.66          |
| ONNX Runtime (INT8)       | 13.77              | 14.96            | 11.23          |

#### MLOps Observations

When reviewing these simulation metrics, several crucial deployment realities surface:

1. **Storage Constraint Unlocking**: Dynamic quantization mathematically compresses the physical model footprint by roughly **4.0x** (from 45 MB down to 11 MB). This solves arguably the biggest hurdle for edge devices: tight RAM ceilings and payload flash capacities.
2. **The "Dynamic Execution" Tax**: You'll notice a classic architectural paradox in emulation—the INT8 model executes *slower* than the FP32 ONNX Runtime on a standard laptop CPU. This happens because dynamic quantization requires continuously casting activation states on-the-fly (converting FP32 activations into bounded INT8 tensors, doing the matrix math, and then inflating the answer back to FP32 for the next layer). Standard x86 processors do not possess native integer accelerators (like VNNI), making this repetitive type-casting slower than just using highly-optimized floating point vectors natively.
3. **Edge Translation**: If we take this exact ONNX INT8 artifact and pass it directly to specialized edge hardware—like an Nvidia Jetson tensor core via TensorRT, or a specialized Coral Edge TPU utilizing offline static-calibration—hardware integer math seamlessly takes over. Once migrated from this emulator sandbox to true edge silicon, the INT8 graphs radically outperform FP32 frameworks on both dimensions simultaneously (speed and footprint).
4. **PyTorch Exporter Incompatibilities**: During framework integration, we encountered severe upstream shape inference bugs (`[ShapeInferenceError] Inferred shape and existing shape differ`) native to PyTorch's default Dynamo ONNX exporter when interpreting flattened CNN feature maps under dynamic configurations. To circumvent this and ensure deployment stability, we locked the exporter graph strictly to `opset_version=18`, swapped uncompliant architectures, and injected explicit `quant_pre_process` cycles from ONNXRuntime to repair geometric mismatches before subsequent downstream executions.

## MLOps & Fleet Telemetry

Monitoring model performance post-deployment is a crucial hurdle for autonomous robotics. The module `src/shadow_mode_telemetry.py` introduces a simulated **Shadow Mode** runtime where an experimental perception model operates silently in the background alongside the active production model during flight.
* **Bandwidth-Aware Packaging**: By recognizing that raw visual edge-to-cloud streams are prohibitively bandwidth-constrained via radio links, the drone actively clears and discards nominal cyclic buffers locally. It exclusively compresses and transmits a lightweight JSON "Flight Event" only when intervention thresholds are breached.
* **Intelligent Disparity Triggers**: The telemetry pipeline organically scrubs the datastream, triggering upstream datalink transmission flags conditionally if: (a) the human pilot seizes manual control, (b) the stable production backbone logs drastic confidence degradation (`< 0.75`), or (c) the experimental shadow architecture fundamentally disagrees with the production baseline prediction by over `20%`.

## Workspace Structure

- `/data`: Raw and processed multi-modal datasets.
- `/models`: Checkpoints, ONNX exports, and saved weights.
- `/scripts`: Utility scripts including our edge model export pipeline.
- `/src`: Core deep perception mechanisms.

---
*Created as the foundation for exploring next-generation robot perception and spatial intelligence.*
