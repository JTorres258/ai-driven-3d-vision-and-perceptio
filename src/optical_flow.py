"""
Optical Flow Estimation Module.

This module provides a robust optical flow estimator utilizing the RAFT (Recurrent All-Pairs Field Transforms)
architecture for accurate pixel-wise motion estimation in video streams.
"""

import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from typing import Tuple, Optional

class FlowEstimator:
    """
    Estimator for optical flow using the RAFT large model.
    """
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        """
        Initializes the FlowEstimator with the pretrained RAFT-large model.
        
        Args:
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=self.weights, progress=False).to(self.device)
        self.model.eval()
        self.transforms = self.weights.transforms()

    def estimate(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """
        Estimates the optical flow between two consecutive image frames.
        
        Args:
            image1 (torch.Tensor): First image frame tensor of shape (B, C, H, W).
            image2 (torch.Tensor): Second image frame tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Estimated optical flow tensor of shape (B, 2, H, W).
        """
        image1, image2 = self.transforms(image1, image2)
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)
        
        with torch.no_grad():
            list_of_flows = self.model(image1, image2)
            # RAFT returns a list of flow predictions; the last one is the final prediction
            predicted_flow = list_of_flows[-1]
            
        return predicted_flow
