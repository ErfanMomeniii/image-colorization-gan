"""
PatchGAN Discriminator for Image Colorization

Architecture based on pix2pix paper:
- 5 convolutional layers
- Takes concatenated L and AB channels as input
- Outputs a grid of predictions (not a single value)
- 70x70 receptive field classifies image patches

PatchGAN focuses on high-frequency structure (texture, local patterns).
The L1 loss handles low-frequency content (overall color).
"""

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator for image colorization.

    Takes concatenated L and AB channels as input.
    Outputs a grid of predictions (real/fake) for patches of the input.
    This allows the discriminator to focus on local texture and style.

    Input:
        - L channel: (batch, 1, 256, 256)
        - AB channels: (batch, 2, 256, 256)
    Output: (batch, 1, 30, 30) - 30x30 grid of patch predictions

    Each output value classifies a 70x70 patch of the input image.
    Total Parameters: ~2.8M
    """

    def __init__(self, in_channels=3, features=64):
        """
        Args:
            in_channels: Number of input channels (L=1 + AB=2 = 3)
            features: Base number of features
        """
        super().__init__()

        # in_channels = L (1) + AB (2) = 3
        self.model = nn.Sequential(
            # Layer 1: No normalization (per pix2pix paper)
            # 256 -> 128
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 128 -> 64
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 64 -> 32
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 32 -> 31 (stride=1)
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer: 31 -> 30 (stride=1)
            # No sigmoid - using BCEWithLogitsLoss
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, L, AB):
        """
        Forward pass through discriminator.

        Args:
            L: Grayscale input (batch, 1, H, W)
            AB: Color channels (batch, 2, H, W) - either real or generated
        Returns:
            Patch-wise predictions (batch, 1, H', W') where H'=W'=30 for 256x256 input
        """
        x = torch.cat([L, AB], dim=1)
        return self.model(x)


def count_parameters(model):
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test discriminator
    disc = PatchDiscriminator()
    L = torch.randn(1, 1, 256, 256)
    AB = torch.randn(1, 2, 256, 256)
    out = disc(L, AB)
    print(f"L shape: {L.shape}")
    print(f"AB shape: {AB.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Total parameters: {count_parameters(disc):,}")
