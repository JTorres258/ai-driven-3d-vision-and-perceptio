"""
Sensor Fusion Module.

Implements multi-modal sensor fusion algorithms, specifically targeting Bird's Eye View (BEV) 
representations by combining RGB cameras and thermal imaging for robust perception.
"""

import torch
import torch.nn as nn
from typing import Tuple

class BEVFusion(nn.Module):
    """
    Bird's-Eye View (BEV) Fusion mechanism combining RGB and Thermal inputs
    using a learned cross-modal attention module.
    """
    def __init__(self, feature_dim: int = 256) -> None:
        """
        Initializes the BEVFusion module.
        
        Args:
            feature_dim (int): The embedding dimension for attention computation.
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        # Dummy linear layers to project inputs to queries, keys, and values
        self.rgb_proj = nn.Linear(feature_dim, feature_dim)
        self.thermal_proj = nn.Linear(feature_dim, feature_dim)
        
        # Cross-modal Attention: Thermal queries RGB info (or vice-versa)
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8, batch_first=True)
        
        # Output projection
        self.fusion_out = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, rgb_features: torch.Tensor, thermal_features: torch.Tensor) -> torch.Tensor:
        """
        Fuses modalities using a dummy attention mechanism.
        
        Args:
            rgb_features (torch.Tensor): Extracted features from RGB camera (B, N, C).
            thermal_features (torch.Tensor): Extracted features from thermal sensor (B, N, C).
            
        Returns:
            torch.Tensor: Fused BEV feature representation (B, N, C).
        """
        # Project features
        rgb_proj = self.rgb_proj(rgb_features)
        thermal_proj = self.thermal_proj(thermal_features)
        
        # Apply attention (e.g., thermal representation querying RGB context)
        # Query: thermal, Key: rgb, Value: rgb
        attn_out, _ = self.attention(query=thermal_proj, key=rgb_proj, value=rgb_proj)
        
        # Concatenate and project to form the final fused feature
        combined = torch.cat([thermal_proj, attn_out], dim=-1)
        fused_features = self.fusion_out(combined)
        
        return fused_features
