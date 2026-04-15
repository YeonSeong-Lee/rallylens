"""TrackNetV3 model architecture for shuttlecock detection.

Reference: https://github.com/qaz812345/TrackNetV3

Two-stage pipeline:
  1. TrackNet   — 2D U-Net that detects shuttlecock heatmaps from a sequence of frames.
  2. InpaintNet — 1D U-Net that refines predicted trajectories by inpainting occluded frames.

Input resolution: 288 × 512  (H × W)
Default sequence length: 8 frames
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class Conv2DBlock(nn.Module):
    """Conv2d (same padding) + BatchNorm + ReLU."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding="same", bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Double2DConv(nn.Module):
    """Two consecutive Conv2DBlocks."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(Conv2DBlock(in_dim, out_dim), Conv2DBlock(out_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Triple2DConv(nn.Module):
    """Three consecutive Conv2DBlocks."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Conv2DBlock(in_dim, out_dim),
            Conv2DBlock(out_dim, out_dim),
            Conv2DBlock(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# TrackNet  (2D U-Net for heatmap detection)
# ---------------------------------------------------------------------------


class TrackNet(nn.Module):
    """2D U-Net that predicts per-frame shuttlecock heatmaps.

    Args:
        in_dim:  Number of input channels.  Default ``3 * seq_len`` (RGB × frames).
        out_dim: Number of output heatmaps. Typically equal to ``seq_len``.

    Input shape:  ``(N, in_dim, 288, 512)``
    Output shape: ``(N, out_dim, 288, 512)`` — sigmoid-activated heatmaps in [0, 1]
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        # Encoder
        self.down_block_1 = Double2DConv(in_dim, 64)    # → (N,  64, 288, 512)
        self.down_block_2 = Double2DConv(64, 128)        # → (N, 128, 144, 256)
        self.down_block_3 = Triple2DConv(128, 256)       # → (N, 256,  72, 128)
        self.bottleneck = Triple2DConv(256, 512)         # → (N, 512,  36,  64)

        # Decoder (with skip connections)
        self.up_block_1 = Triple2DConv(512 + 256, 256)  # skip from down_block_3
        self.up_block_2 = Double2DConv(256 + 128, 128)  # skip from down_block_2
        self.up_block_3 = Double2DConv(128 + 64, 64)    # skip from down_block_1

        self.predictor = nn.Conv2d(64, out_dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self._pool = nn.MaxPool2d(2, 2)
        self._up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down_block_1(x)           # (N,  64, 288, 512)
        x2 = self.down_block_2(self._pool(x1))  # (N, 128, 144, 256)
        x3 = self.down_block_3(self._pool(x2))  # (N, 256,  72, 128)
        x = self.bottleneck(self._pool(x3))      # (N, 512,  36,  64)

        x = self.up_block_1(torch.cat([self._up(x), x3], dim=1))   # (N, 256,  72, 128)
        x = self.up_block_2(torch.cat([self._up(x), x2], dim=1))   # (N, 128, 144, 256)
        x = self.up_block_3(torch.cat([self._up(x), x1], dim=1))   # (N,  64, 288, 512)

        return self.sigmoid(self.predictor(x))  # (N, out_dim, 288, 512)


# ---------------------------------------------------------------------------
# InpaintNet  (1D U-Net for trajectory refinement)
# ---------------------------------------------------------------------------


class Conv1DBlock(nn.Module):
    """Conv1d (same padding) + LeakyReLU."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding="same", bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


class Double1DConv(nn.Module):
    """Two consecutive Conv1DBlocks."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(Conv1DBlock(in_dim, out_dim), Conv1DBlock(out_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InpaintNet(nn.Module):
    """1D U-Net that inpaints occluded shuttlecock positions in a trajectory.

    Takes a predicted coordinate sequence and a binary mask indicating which
    frames need inpainting, then outputs a refined coordinate sequence.

    Args:
        x: ``(N, L, 2)`` — predicted (x, y) coordinates, normalised to [0, 1]
        m: ``(N, L, 1)`` — inpaint mask  (1 = needs refinement, 0 = keep as-is)

    Returns: ``(N, L, 2)`` — refined coordinates in [0, 1]
    """

    def __init__(self) -> None:
        super().__init__()
        # Encoder
        self.down_1 = Conv1DBlock(3, 32)
        self.down_2 = Conv1DBlock(32, 64)
        self.down_3 = Conv1DBlock(64, 128)
        self.bottleneck = Double1DConv(128, 256)

        # Decoder (with skip connections)
        self.up_1 = Conv1DBlock(256 + 128, 128)
        self.up_2 = Conv1DBlock(128 + 64, 64)
        self.up_3 = Conv1DBlock(64 + 32, 32)

        self.predictor = nn.Conv1d(32, 2, kernel_size=3, padding="same")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # x: (N, L, 2), m: (N, L, 1)
        seq = torch.cat([x, m], dim=2).permute(0, 2, 1)  # (N, 3, L)

        s1 = self.down_1(seq)        # (N,  32, L)
        s2 = self.down_2(s1)         # (N,  64, L)
        s3 = self.down_3(s2)         # (N, 128, L)
        s = self.bottleneck(s3)      # (N, 256, L)

        s = self.up_1(torch.cat([s, s3], dim=1))  # (N, 128, L)
        s = self.up_2(torch.cat([s, s2], dim=1))  # (N,  64, L)
        s = self.up_3(torch.cat([s, s1], dim=1))  # (N,  32, L)

        out = self.sigmoid(self.predictor(s))  # (N, 2, L)
        return out.permute(0, 2, 1)            # (N, L, 2)
