"""
Voice CNN Model - Binary Classification (Human vs AI Voice)
Architecture:
    Input: (batch, 1, 40, 300) - MFCC features
    ↓
    Conv2d(1 → 16) + BatchNorm + ReLU + MaxPool2d
    ↓
    Conv2d(16 → 32) + BatchNorm + ReLU + MaxPool2d
    ↓
    Conv2d(32 → 64) + BatchNorm + ReLU + MaxPool2d
    ↓
    Global Average Pooling → (batch, 64)
    ↓
    Linear(64 → 2) - Binary Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoiceCNN(nn.Module):
    """CNN for detecting AI vs Human voice"""
    
    def __init__(self):
        super().__init__()

        # Conv Block 1: 1 → 16 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=0),
            nn.BatchNorm2d(16),
        )
        
        # Conv Block 2: 16 → 32 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
        )
        
        # Conv Block 3: 32 → 64 channels
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
        )
        
        # Activation and pooling
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head: 64 → 2 classes
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: (batch, 1, 40, 300) MFCC features
        Returns:
            logits: (batch, 2) classification logits
        """
        # Block 1: Conv(1→16) + BatchNorm + ReLU + MaxPool
        x = self.block1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Block 2: Conv(16→32) + BatchNorm + ReLU + MaxPool
        x = self.block2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Block 3: Conv(32→64) + BatchNorm + ReLU + MaxPool
        x = self.block3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Global Average Pooling: (batch, 64, H, W) → (batch, 64)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Classification: 64 → 2
        x = self.fc(x)
        
        return x
