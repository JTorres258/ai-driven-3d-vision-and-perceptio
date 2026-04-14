"""
Monocular Depth Estimation Module.

This module leverages the MiDaS model to infer high-quality relative depth maps from 
single RGB images, aiding in 3D scene perception for drone navigation.
"""

import torch
from typing import Any
import numpy as np
import cv2

class DepthEstimator:
    """
    Monocular depth estimator using the MiDaS small model for efficient inference.
    """
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        """
        Initializes the DepthEstimator using the Torch Hub MiDaS implementation.
        
        Args:
            device (str): Device to run the depth estimation on ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        # Load the MiDaS small model
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Load transforms to resize and normalize the image
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = midas_transforms.small_transform

    def estimate(self, rgb_image: np.ndarray) -> torch.Tensor:
        """
        Estimates the depth map from an RGB image.
        
        Args:
            rgb_image (np.ndarray): Input RGB image in HWC format (e.g., from OpenCV).
            
        Returns:
            torch.Tensor: The predicted depth map of shape (1, H, W).
        """
        input_batch = self.transform(rgb_image).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

            # Resize the prediction to match the original image resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb_image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)

        return prediction
