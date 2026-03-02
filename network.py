"""
Neural network for Quoridor (AlphaZero-style with Squeeze-and-Excitation).

Architecture:
    Input  -> Conv 3x3 -> BN -> ReLU
           -> N residual blocks with SE attention
    Policy -> Conv 1x1 -> BN -> ReLU -> FC -> 140 logits (masked)
    Value  -> Conv 1x1 -> BN -> ReLU -> FC -> FC -> tanh -> scalar

8 blocks x 128 channels ~ 1.5M params. Sized for Apple Silicon.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: channel attention via global average pooling."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = x.mean(dim=(2, 3))             # (B, C)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.view(b, c, 1, 1)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual)


class QuoridorNet(nn.Module):
    def __init__(self, input_planes: int = 9, num_blocks: int = 8,
                 num_channels: int = 128, action_size: int = 140):
        super().__init__()
        self.action_size = action_size

        # Input block
        self.conv_in = nn.Conv2d(input_planes, num_channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_channels) for _ in range(num_blocks)]
        )

        # Policy head
        ph_channels = 4
        self.policy_conv = nn.Conv2d(num_channels, ph_channels, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(ph_channels)
        self.policy_fc = nn.Linear(ph_channels * 9 * 9, action_size)

        # Value head
        vh_channels = 2
        self.value_conv = nn.Conv2d(num_channels, vh_channels, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(vh_channels)
        self.value_fc1 = nn.Linear(vh_channels * 9 * 9, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor, legal_mask: torch.Tensor = None):
        """
        Args:
            state: (B, C_in, 9, 9)
            legal_mask: (B, 140) binary mask, 1=legal

        Returns:
            log_policy: (B, 140) log-probabilities (masked)
            value: (B, 1) in [-1, 1]
        """
        x = F.relu(self.bn_in(self.conv_in(state)))
        x = self.res_blocks(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        if legal_mask is not None:
            p = p.masked_fill(legal_mask == 0, -1e8)
        log_policy = F.log_softmax(p, dim=1)

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return log_policy, v

    @torch.no_grad()
    def predict(self, state_np: np.ndarray, mask_np: np.ndarray,
                device: torch.device) -> tuple:
        """Single-sample inference for MCTS.

        Returns:
            policy: np.ndarray (140,) probabilities
            value: float scalar
        """
        self.eval()
        state = torch.from_numpy(state_np).unsqueeze(0).to(device)
        mask = torch.from_numpy(mask_np).unsqueeze(0).to(device)
        log_p, v = self(state, mask)
        policy = torch.exp(log_p).squeeze(0).cpu().numpy()
        value = v.item()
        return policy, value


if __name__ == "__main__":
    from config import Config
    cfg = Config()
    net = QuoridorNet(cfg.input_planes, cfg.num_res_blocks,
                      cfg.num_channels, cfg.action_size)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Parameters: {n_params:,}")

    # Smoke test forward pass
    dummy_state = torch.randn(2, 9, 9, 9)
    dummy_mask = torch.ones(2, 140)
    log_p, v = net(dummy_state, dummy_mask)
    print(f"Policy shape: {log_p.shape}  Value shape: {v.shape}")
    print(f"Policy sums to ~1: {torch.exp(log_p[0]).sum().item():.4f}")
    print(f"Value range: [{v.min().item():.3f}, {v.max().item():.3f}]")
