#!/usr/bin/env python3
"""
Feature extractors for the RuneScape environment.

This module defines classes for extracting features from the game state
for use with neural networks.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict


class CombinedExtractor(BaseFeaturesExtractor):
    """Feature extractor that processes both image and vector observations."""
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        """Initialize the feature extractor.
        
        Args:
            observation_space: The observation space (Dict with 'image' and 'vector' keys)
            features_dim: The dimension of the output features
        """
        super().__init__(observation_space, features_dim)
        
        # Extract shapes from observation space
        image_shape = observation_space.spaces['image'].shape
        vector_shape = observation_space.spaces['vector'].shape
        
        # Calculate vector dimension
        vector_dim = vector_shape[0] if len(vector_shape) > 0 else 0
        
        # Determine image channels (assume last dim is channels for [H,W,C] format)
        channels = 3  # Default to RGB
        if len(image_shape) == 3:
            channels = image_shape[2] if image_shape[2] in [1, 3, 4] else image_shape[0]
        
        # CNN for processing images
        cnn_output_dim = 128
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute CNN output size by doing one forward pass
        with torch.no_grad():
            sample_img = torch.zeros(1, channels, 84, 84)
            n_flatten = self.cnn(sample_img).shape[1]
        
        # Linear layer after CNN
        self.cnn_linear = nn.Sequential(
            nn.Linear(n_flatten, cnn_output_dim),
            nn.ReLU(),
        )
        
        # MLP for processing vector data
        vector_output_dim = 128
        self.mlp = nn.Sequential(
            nn.Linear(vector_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, vector_output_dim),
            nn.ReLU(),
        )
        
        # Final layer to combine both features
        self.final = nn.Linear(cnn_output_dim + vector_output_dim, features_dim)
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process the observation dictionary.
        
        Args:
            observations: Dictionary with 'image' and 'vector' keys
            
        Returns:
            torch.Tensor: The extracted features
        """
        # Process image
        # Ensure image is in right format [B, C, H, W] and normalized
        if observations['image'].shape[-1] == 3:  # If in format [B, H, W, C]
            img = observations['image'].permute(0, 3, 1, 2)
        else:
            img = observations['image']
        
        img = img.float() / 255.0  # Normalize to [0, 1]
        cnn_features = self.cnn_linear(self.cnn(img))
        
        # Process vector
        vector_features = self.mlp(observations['vector'])
        
        # Combine features
        combined = torch.cat([cnn_features, vector_features], dim=1)
        return self.final(combined) 