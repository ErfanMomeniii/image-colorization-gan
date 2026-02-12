"""
U-Net Generator for Image Colorization

Architecture based on pix2pix paper:
- Encoder: 7 downsampling blocks with stride-2 convolutions
- Bottleneck: Single convolution layer
- Decoder: 7 upsampling blocks with skip connections
- Output: Tanh activation for AB channels

Skip connections preserve spatial information lost during downsampling.
"""

import torch
import torch.nn as nn


class UNetDownBlock(nn.Module):
    """
    Downsampling block for U-Net encoder.

    Structure: Conv2d -> BatchNorm (optional) -> LeakyReLU -> Dropout (optional)
    Reduces spatial dimensions by factor of 2.
    """

    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            normalize: Whether to apply batch normalization
            dropout: Dropout probability (0.0 = no dropout)
        """
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetUpBlock(nn.Module):
    """
    Upsampling block for U-Net decoder with skip connections.

    Structure: ConvTranspose2d -> BatchNorm -> ReLU -> Dropout (optional) -> Concat(skip)
    Increases spatial dimensions by factor of 2.
    """

    def __init__(self, in_channels, out_channels, dropout=0.0):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dropout: Dropout probability (0.0 = no dropout)
        """
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        """
        Args:
            x: Input tensor from previous decoder layer
            skip: Skip connection tensor from corresponding encoder layer
        Returns:
            Concatenated output tensor
        """
        x = self.block(x)
        x = torch.cat([x, skip], dim=1)
        return x


class UNetGenerator(nn.Module):
    """
    U-Net based generator for image colorization.

    Takes L channel (grayscale) as input and outputs AB channels.
    Architecture follows pix2pix paper with skip connections.

    Input: (batch, 1, 256, 256) - L channel normalized to [-1, 1]
    Output: (batch, 2, 256, 256) - AB channels normalized to [-1, 1]

    Total Parameters: ~54.4M
    """

    def __init__(self, in_channels=1, out_channels=2, features=64):
        """
        Args:
            in_channels: Number of input channels (1 for L channel)
            out_channels: Number of output channels (2 for AB channels)
            features: Base number of features (doubled at each encoder layer)
        """
        super().__init__()

        # Encoder (downsampling) - reduces from 256 to 1
        self.down1 = UNetDownBlock(in_channels, features, normalize=False)  # 256 -> 128
        self.down2 = UNetDownBlock(features, features * 2)                   # 128 -> 64
        self.down3 = UNetDownBlock(features * 2, features * 4)               # 64 -> 32
        self.down4 = UNetDownBlock(features * 4, features * 8)               # 32 -> 16
        self.down5 = UNetDownBlock(features * 8, features * 8)               # 16 -> 8
        self.down6 = UNetDownBlock(features * 8, features * 8)               # 8 -> 4
        self.down7 = UNetDownBlock(features * 8, features * 8)               # 4 -> 2

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, kernel_size=4, stride=2, padding=1),  # 2 -> 1
            nn.ReLU(inplace=True),
        )

        # Decoder (upsampling with skip connections) - expands from 1 to 256
        self.up1 = UNetUpBlock(features * 8, features * 8, dropout=0.5)      # 1 -> 2
        self.up2 = UNetUpBlock(features * 16, features * 8, dropout=0.5)     # 2 -> 4
        self.up3 = UNetUpBlock(features * 16, features * 8, dropout=0.5)     # 4 -> 8
        self.up4 = UNetUpBlock(features * 16, features * 8)                  # 8 -> 16
        self.up5 = UNetUpBlock(features * 16, features * 4)                  # 16 -> 32
        self.up6 = UNetUpBlock(features * 8, features * 2)                   # 32 -> 64
        self.up7 = UNetUpBlock(features * 4, features)                       # 64 -> 128

        # Final layer - outputs AB channels
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.Tanh(),  # Output in range [-1, 1]
        )

    def forward(self, x):
        """
        Forward pass through U-Net.

        Args:
            x: Input L channel tensor (batch, 1, H, W)
        Returns:
            Predicted AB channels (batch, 2, H, W)
        """
        # Encoder - save outputs for skip connections
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        # Bottleneck
        bottleneck = self.bottleneck(d7)

        # Decoder with skip connections
        u1 = self.up1(bottleneck, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


def count_parameters(model):
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test generator
    gen = UNetGenerator()
    x = torch.randn(1, 1, 256, 256)
    out = gen(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Total parameters: {count_parameters(gen):,}")
