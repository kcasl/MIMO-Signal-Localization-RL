"""SAVN-CE/MAGNet-style modality encoders plus a sliding observation memory.

Mirrors MAGNet's PoseEncoder / ActionEncoder / AudioEncoder (Conv-BN-ReLU-Pool-
Dropout, then a linear embed) and SAVNCE_StateEncoder (fusion MLP + Transformer
over an external episode memory). MIMO CSI is the analogue of the spectrogram:
a 4×4 amplitude/phase grid instead of a time-frequency map.

The memory bank stores *encoder* outputs, the DAgger-era sliding history of
raw frames but after each modality has been embedded.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from rf_physics import COLLISION_DIM, CSI_BASE_DIM, N_RX, N_TX, RF_PATH_DIM

CSI_GRID = N_RX * N_TX  # 16
CSI_SVD = N_RX
CSI_POWER = 1
MIMO_RAW_DIM = CSI_BASE_DIM + RF_PATH_DIM + COLLISION_DIM  # 51
POSE_DIM = 5
PREV_ACTION_DIM = 2
FEATURE_LAYOUT = (MIMO_RAW_DIM, PREV_ACTION_DIM, POSE_DIM)


class ConvBlock(nn.Module):
    """MAGNet AudioEncoder block: conv → norm → ReLU → pool → dropout."""

    def __init__(self, in_channels: int, out_channels: int, pool_size=None, dropout: float = 0.1):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.pool = nn.Identity() if pool_size is None else nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm(self.conv(x)))
        x = self.pool(x)
        return self.dropout(x)


class MIMOChannelEncoder(nn.Module):
    """CRNN-style encoder for the 4×4 MIMO channel, MAGNet AudioEncoder layout.

    Amplitude and phase are a 2-channel 4×4 grid. SVD, mean power, multipath
    RF descriptors, and last-collision are fused after the conv stack.
    """

    def __init__(self, embedding_size: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embedding_size = embedding_size
        self.conv_block1 = ConvBlock(2, 32, pool_size=None, dropout=dropout)
        self.conv_block2 = ConvBlock(32, 64, pool_size=None, dropout=dropout)
        self.conv_block3 = ConvBlock(64, 128, pool_size=2, dropout=dropout)
        self.fc_grid = nn.Linear(128 * 2 * 2, embedding_size)
        aux_dim = CSI_SVD + CSI_POWER + RF_PATH_DIM + COLLISION_DIM
        self.fc_aux = nn.Sequential(nn.Linear(aux_dim, 32), nn.ReLU())
        self.out = nn.Linear(embedding_size + 32, embedding_size)

    def forward(self, mimo: torch.Tensor) -> torch.Tensor:
        amp = mimo[..., :CSI_GRID].reshape(-1, 1, N_RX, N_TX)
        phase = mimo[..., CSI_GRID:2 * CSI_GRID].reshape(-1, 1, N_RX, N_TX)
        grid = self.conv_block3(self.conv_block2(self.conv_block1(torch.cat((amp, phase), 1))))
        spatial = self.fc_grid(grid.flatten(1))
        aux = self.fc_aux(mimo[..., 2 * CSI_GRID:])
        return self.out(torch.cat((spatial, aux), dim=-1))


class PoseEncoder(nn.Module):
    """MAGNet PoseEncoder: Linear(5 → embedding) on already-formatted pose."""

    def __init__(self, pose_dim: int = POSE_DIM, embedding_size: int = 16):
        super().__init__()
        self.embedding_size = embedding_size
        self.pose_encoder = nn.Linear(pose_dim, embedding_size)

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        return self.pose_encoder(pose)


class ActionEncoder(nn.Module):
    """MAGNet ActionEncoder analogue for a continuous previous heading (cos, sin)."""

    def __init__(self, action_dim: int = PREV_ACTION_DIM, embedding_size: int = 16):
        super().__init__()
        self.embedding_size = embedding_size
        self.action_encoder = nn.Linear(action_dim, embedding_size)

    def forward(self, prev_action: torch.Tensor) -> torch.Tensor:
        return self.action_encoder(prev_action)


class SceneMemoryEncoder(nn.Module):
    """SAVNCE_StateEncoder: fuse tokens, attend current query over episode memory."""

    def __init__(
        self,
        feature_size: int,
        embedding_size: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.fusion_encoder = nn.Sequential(
            nn.Linear(feature_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size),
        )
        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=embedding_size,
            dropout=dropout,
            activation="relu",
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor, memory_masks: torch.Tensor) -> torch.Tensor:
        """x: (N, F); memory: (M, N, F); memory_masks: (N, M) with 1 = valid."""
        memory_masks = torch.cat(
            [memory_masks, torch.ones(memory_masks.shape[0], 1, device=memory_masks.device)],
            dim=1,
        )
        tokens = torch.cat([memory, x.unsqueeze(0)])
        m_len, batch = tokens.shape[:2]
        tokens = self.fusion_encoder(tokens.reshape(m_len * batch, -1)).view(m_len, batch, -1)
        pad = (1.0 - memory_masks) > 0
        attended = self.transformer(
            tokens,
            tokens[-1:],
            src_key_padding_mask=pad,
            memory_key_padding_mask=pad,
        )
        return attended[-1]


def split_observation(observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mimo_dim, prev_dim, _ = FEATURE_LAYOUT
    mimo = observation[..., :mimo_dim]
    prev = observation[..., mimo_dim:mimo_dim + prev_dim]
    pose = observation[..., mimo_dim + prev_dim:]
    return mimo, prev, pose
