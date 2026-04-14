# AI-Driven 3D Vision & Perception for Autonomous Flight

Welcome to the **Drone Perception Workspace**, a cutting-edge repository geared towards the seamless integration of deeply learned visual primitives into autonomous navigation and path-planning workflows.

## Background & Philosophy

Driven by **Jorge García-Torres Fernández's** extensive background bridging Ph.D. research with production-ready Computer Vision, this project represents a fundamental transition from classical, heuristic-driven C++ CV pipelines to end-to-end Deep SLAM and multi-modal perception systems. As modern edge hardware capabilities expand, offloading classical geometric formulations to learned deep representations—such as depth and flow estimators—has become critical for scalable, resilient autonomous flight.

## Core Modules

* **Optical Flow (`src/optical_flow.py`)**: Motion estimation powered by RAFT (Recurrent All-Pairs Field Transforms) to robustly track high-velocity dynamics.
* **Monocular Depth (`src/depth_estimation.py`)**: Deep relative depth estimation leveraging the MiDaS architecture to replace or augment active RGB-D sensors in open-air flight.
* **Sensor Fusion (`src/sensor_fusion.py`)**: A Bird's Eye View (BEV) multi-head attention module designed to fuse continuous RGB streams with thermal imaging, yielding robust perception in visually degraded environments.

## Deployment to Edge

Our Deep CV modules must run with minimal latency onboard compute-constrained drones (e.g., Jetson platforms). The script `scripts/export_to_edge.py` facilitates exporting our PyTorch perception backbones directly into optimized ONNX formats with dynamic tensor resolutions, priming them for inference accelerators like TensorRT.

## Workspace Structure

- `/applications`: High-level application logic spanning navigation and UI systems.
- `/data`: Raw and processed multi-modal datasets.
- `/models`: Checkpoints, ONNX exports, and saved weights.
- `/notebooks`: Interactive Jupyter environments for exploratory data analysis and visual debugging.
- `/scripts`: Utility scripts including our edge model export pipeline.
- `/src`: Core deep perception mechanisms.

---
*Created as the foundation for exploring next-generation robot perception and spatial intelligence.*
