"""The CNN client's policy/score network, as a trainable module.

Same layers and parameter names as the TorchScript `traced.pt` the client loads, so
`load_traced` copies its weights and `export` writes a file the client can load. A traced
file can't be fine-tuned directly: its batch norms were traced in eval mode.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .encoding import HEIGHT, N_ACTIONS, N_CHANNEL, N_SCORE, WIDTH


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class Bias(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        return x + self.bias


class PolicyValueNetwork(nn.Module):
    def __init__(self, channels: int = 64, blocks: int = 12):
        super().__init__()
        self.conv1 = nn.Conv2d(N_CHANNEL, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        # policy: (C, 32, 16) -> upsample width -> (C, 32, 32) -> (2, 16, 32) = rotation x speed x angle
        self.upsample1 = nn.Upsample(scale_factor=(1, 2))
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=(2, 1), stride=(2, 1), bias=False)
        self.policy_bias = Bias(N_ACTIONS)
        # score: distribution of the hammer team's net points in this end
        self.score_conv1 = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.score_norm1 = nn.BatchNorm2d(2)
        self.score_fc1 = nn.Linear(2 * HEIGHT * WIDTH, 256)
        self.score_fc2 = nn.Linear(256, N_SCORE)

    def forward(self, x):
        h = self.blocks(F.relu(self.norm1(self.conv1(x))))
        policy = self.policy_bias(torch.flatten(self.policy_conv(self.upsample1(h)), 1))
        s = torch.flatten(F.relu(self.score_norm1(self.score_conv1(h))), 1)
        score = self.score_fc2(F.relu(self.score_fc1(s)))
        return policy, score


def load_traced(path: str | Path, map_location="cpu") -> PolicyValueNetwork:
    """Network with the weights of a client `traced.pt`."""
    traced = torch.jit.load(str(path), map_location=map_location)
    model = PolicyValueNetwork()
    model.load_state_dict(traced.state_dict())
    return model


def load(path: str | Path, map_location="cpu") -> PolicyValueNetwork:
    """From a `traced.pt` or a training checkpoint (`checkpoint-*.pth`)."""
    path = Path(path)
    if path.suffix == ".pth":
        model = PolicyValueNetwork()
        model.load_state_dict(torch.load(path, map_location=map_location)["model"])
        return model
    return load_traced(path, map_location)


@torch.no_grad()
def export(model: PolicyValueNetwork, path: str | Path) -> None:
    """Trace in eval mode on CPU and save a `traced.pt` for the client; checks the round trip."""
    was_training = model.training
    model = model.eval()
    device = next(model.parameters()).device
    cpu = model.to("cpu")
    example = torch.zeros((2, N_CHANNEL, HEIGHT, WIDTH))
    example[:, 2] = 1.0
    traced = torch.jit.trace(cpu, example)
    traced.save(str(path))
    reloaded = torch.jit.load(str(path))
    x = torch.rand((4, N_CHANNEL, HEIGHT, WIDTH))
    for a, b in zip(cpu(x), reloaded(x)):
        torch.testing.assert_close(a, b)
    model.to(device)
    model.train(was_training)
