import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import Config

class ConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = F.relu(out, inplace=True)
        return out

class GomokuNetDyn(nn.Module):
    def __init__(self, channels=64, num_res_blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.resblocks = nn.Sequential(*[ConvBlock(channels) for _ in range(num_res_blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=1)
        )
        self.value_head_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
        )
        self.value_fc1 = nn.Linear(channels // 2, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        out = self.stem(x)
        out = self.resblocks(out)
        policy = self.policy_head(out)
        b, _, h, w = policy.size()
        policy_logits = policy.view(b, h * w)
        v = self.value_head_conv(out)
        v = F.adaptive_avg_pool2d(v, (1, 1))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v), inplace=True)
        value = self.value_fc2(v).squeeze(-1)
        return policy_logits, value

def load_model_if_exists(model, file_path):
    if os.path.exists(file_path):
        try:
            state = torch.load(file_path, map_location=torch.device('cpu'))
            model.load_state_dict(state)
            print(f"Loaded model weights from {file_path}")
            return True
        except Exception as e:
            print(f"Failed to load model from {file_path}: {e}")
            return False
    else:
        print(f"No saved model weights found at {file_path}")
        return False

