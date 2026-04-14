"""
Edge Export Script.

Utility script for exporting PyTorch models to ONNX format, optimizing them 
for inference on edge devices (like Jetson or edge TPUs) onboard the drone.
"""

import torch
import torch.nn as nn
from pathlib import Path
import argparse

class DummyModel(nn.Module):
    """
    A minimal model representing a generic perception backbone to demonstrate ONNX export.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def export_model(output_path: str = "models/model.onnx") -> None:
    """
    Exports a PyTorch model to ONNX with dynamic axes for variable batch sizes
    and image dimensions.
    
    Args:
        output_path (str): The destination file path for the exported ONNX model.
    """
    print("Initialize model for export...")
    model = DummyModel()
    model.eval()

    # Dummy input representing a batch of RGB images (Batch, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Define dynamic axes to allow variable-sized inputs
    dynamic_axes = {
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size'}
    }

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,                         # model being run
        dummy_input,                   # model input
        output_path,                   # where to save the model
        export_params=True,            # store the trained parameter weights inside the model file
        opset_version=14,              # the ONNX version to export the model to
        do_constant_folding=True,      # whether to execute constant folding for optimization
        input_names=['input'],         # the model's input names
        output_names=['output'],       # the model's output names
        dynamic_axes=dynamic_axes      # variable length axes
    )
    print("Export completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Model to ONNX.")
    parser.add_argument("--output", type=str, default="models/model.onnx", help="Output path for ONNX file.")
    args = parser.parse_args()
    
    export_model(args.output)
