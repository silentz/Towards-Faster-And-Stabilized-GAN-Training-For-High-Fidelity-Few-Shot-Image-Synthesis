import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class Conv2d(nn.Module):

    def __init__(self, *args, **kwargs):
        self._conv = spectral_norm(
                nn.Conv2d(*args, **kwargs)
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv(input)


class ConvTranspose2d(nn.Module):

    def __init__(self, *args, **kwargs):
        self._conv = spectral_norm(
                nn.ConvTranspose2d(*args, **kwargs)
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv(input)


class Linear(nn.Module):

    def __init__(self, *args, **kwargs):
        self._linear = spectral_norm(
                nn.Linear(*args, **kwargs)
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._linear(input)


class Swish(nn.Module):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sigmoid(input)


class SLEBlock(nn.Module):

    def __init__(self, in_channels: int,
                       out_channels: int):
        super().__init__()

        self._layers = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=4),
                Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=4,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                Swish(),
                Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                nn.Sigmoid(),
            )

    def forward(self, low_dim: torch.Tensor,
                      high_dim: torch.Tensor) -> torch.Tensor:
        return high_dim * self._layers(low_dim)

