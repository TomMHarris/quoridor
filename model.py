import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroNet(nn.Module):
    def __init__(self, board_size=9):
        super().__init__()
        self.size = board_size
        
        # Input: 4 Channels (P1, P2, WallV, WallH)
        self.conv1 = nn.Conv2d(4, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Policy Head
        # Output Action Space:
        # 1 channel for movement (size x size)
        # 1 channel for V-walls (size-1 x size-1) -> padded to size x size
        # 1 channel for H-walls (size-1 x size-1) -> padded to size x size
        # Total output: 3 * size * size flattened
        self.policy_conv = nn.Conv2d(128, 32, 1)
        self.policy_fc = nn.Linear(32 * board_size * board_size, 3 * board_size * board_size)
        
        # Value Head
        self.value_conv = nn.Conv2d(128, 3, 1)
        self.value_fc1 = nn.Linear(3 * board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)
        
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")