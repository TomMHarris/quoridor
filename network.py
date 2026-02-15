"""
Neural network for Quoridor (AlphaZero-style).

Architecture:
    Input  → Conv 3×3 → BN → ReLU
           → N residual blocks (conv-bn-relu-conv-bn + skip)
    Policy → Conv 1×1 → BN → ReLU → FC → 140 logits (masked softmax)
    Value  → Conv 1×1 → BN → ReLU → FC → FC → tanh → scalar

Sized for M1 MacBook Air: 4 blocks × 64 channels ≈ 200K params.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from engine import MoveEncoder


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class QuoridorNet(nn.Module):
    """
    Inputs:
        state:      (B, C_in, 9, 9)  game state tensor
        legal_mask: (B, 140)          1 = legal, 0 = illegal

    Outputs:
        policy: (B, 140)  probability distribution over actions
        value:  (B, 1)    expected outcome from current player's perspective
    """

    ACTION_SIZE = MoveEncoder.TOTAL_ACTIONS  # 140

    def __init__(self, input_planes: int = 7, num_blocks: int = 4,
                 num_channels: int = 64):
        super().__init__()

        # ── Input block ──
        self.input_conv = nn.Conv2d(input_planes, num_channels, 3,
                                    padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_channels)

        # ── Residual tower ──
        self.res_blocks = nn.Sequential(
            *[ResBlock(num_channels) for _ in range(num_blocks)]
        )

        # ── Policy head ──
        self.policy_conv = nn.Conv2d(num_channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 9 * 9, self.ACTION_SIZE)

        # ── Value head ──
        self.value_conv = nn.Conv2d(num_channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 9 * 9, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, state, legal_mask=None):
        # Shared trunk
        x = F.relu(self.input_bn(self.input_conv(state)))
        x = self.res_blocks(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Mask illegal moves (set to -inf before softmax)
        if legal_mask is not None:
            p = p.masked_fill(legal_mask == 0, float("-inf"))

        policy = F.softmax(p, dim=1)

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value

    def predict(self, state_np: np.ndarray, legal_mask_np: np.ndarray,
                device: str = "cpu") -> tuple[np.ndarray, float]:
        """
        Single-state inference for MCTS. Returns (policy, value) as numpy.
        """
        self.eval()
        with torch.no_grad():
            s = torch.from_numpy(state_np).unsqueeze(0).to(device)
            m = torch.from_numpy(legal_mask_np).unsqueeze(0).to(device)
            policy, value = self(s, m)
        return policy.squeeze(0).cpu().numpy(), value.item()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    net = QuoridorNet(input_planes=7, num_blocks=4, num_channels=64)
    print(f"Parameters: {count_parameters(net):,}")

    # Dummy forward pass
    state = torch.randn(8, 7, 9, 9)
    mask = torch.ones(8, 140)
    policy, value = net(state, mask)
    print(f"Policy shape: {policy.shape}")  # (8, 140)
    print(f"Value shape:  {value.shape}")   # (8, 1)
    print(f"Policy sums:  {policy.sum(dim=1)}")  # should be ~1.0
    print(f"Value range:  [{value.min():.3f}, {value.max():.3f}]")

    # Single predict
    s = np.random.randn(7, 9, 9).astype(np.float32)
    m = np.ones(140, dtype=np.float32)
    p, v = net.predict(s, m)
    print(f"\nSingle predict: policy sum={p.sum():.4f}, value={v:.4f}")
