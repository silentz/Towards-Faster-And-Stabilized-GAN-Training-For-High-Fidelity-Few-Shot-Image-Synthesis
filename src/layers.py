import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.utils import spectral_norm


class Conv2d(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._conv = spectral_norm(
                nn.Conv2d(*args, **kwargs)
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv(input)


class ConvTranspose2d(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._conv = spectral_norm(
                nn.ConvTranspose2d(*args, **kwargs)
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv(input)


class Linear(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._linear = spectral_norm(
                nn.Linear(*args, **kwargs)
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._linear(input)


class Noise(nn.Module):

    def __init__(self):
        super().__init__()
        self._weight = nn.Parameter(
                torch.zeros(1),
                requires_grad=True,
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = input.shape
        noise = torch.randn(batch_size, 1, height, width, device=input.device)
        return self._weight * noise + input


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
                nn.SiLU(),
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


class UpsampleBlockT1(nn.Module):

    def __init__(self, in_channels: int,
                       out_channels: int):
        super().__init__()

        self._layers = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels * 2,
                        kernel_size=3,
                        stride=1,
                        padding='same',
                        bias=False,
                    ),
                nn.BatchNorm2d(num_features=out_channels * 2),
                nn.GLU(dim=1),
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._layers(input)


class UpsampleBlockT2(nn.Module):

    def __init__(self, in_channels: int,
                       out_channels: int):
        super().__init__()

        self._layers = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels * 2,
                        kernel_size=3,
                        stride=1,
                        padding='same',
                        bias=False,
                    ),
                Noise(),
                BatchNorm2d(num_features=out_channels * 2),
                nn.GLU(dim=1),
                Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels * 2,
                        kernel_size=3,
                        stride=1,
                        padding='same',
                        bias=False,
                    ),
                Noise(),
                nn.BatchNorm2d(num_features=out_channels * 2),
                nn.GLU(dim=1),
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._layers(input)


class DownsampleBlockT1(nn.Module):

    def __init__(self, in_channels: int,
                       out_channels: int):
        super().__init__()

        self._layers = nn.Sequential(
                Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                nn.BatchNorm2d(num_features=out_channels),
                nn.LeakyReLU(negative_slope=0.2),
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._layers(input)


class DownsampleBlockT2(nn.Module):

    def __init__(self, in_channels: int,
                       out_channels: int):
        super().__init__()

        self._layers_1 = nn.Sequential(
                Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                nn.BatchNorm2d(num_features=out_channels),
                nn.LeakyReLU(negative_slope=0.2),
                Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding='same',
                        bias=False,
                    ),
                nn.BatchNorm2d(num_features=out_channels),
                nn.LeakyReLU(negative_slope=0.2),
            )

        self._layers_2 = nn.Sequential(
                nn.AvgPool2d(
                        kernel_size=2,
                        stride=2,
                    ),
                Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                nn.BatchNorm2d(num_features=out_channels),
                nn.LeakyReLU(negative_slope=0.2),
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        t1 = self._layers_1(input)
        t2 = self._layers_2(input)
        return (t1 + t2) / 2


class Decoder(nn.Module):

    def __init__(self, in_channels: int,
                       out_channels: int):
        super().__init__()

        self._channels = {
                16:   128,
                32:   64,
                64:   64,
                128:  32,
                256:  16,
                512:  8,
                1024: 4,
            }

        self._layers = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=8),
                UpsampleBlockT1(in_channels=in_channels,        out_channels=self._channels[16]),
                UpsampleBlockT1(in_channels=self._channels[16], out_channels=self._channels[32]),
                UpsampleBlockT1(in_channels=self._channels[32], out_channels=self._channels[64]),
                UpsampleBlockT1(in_channels=self._channels[64], out_channels=self._channels[128]),
                Conv2d(
                        in_channels=self._channels[128],
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding='same',
                        bias=False,
                    ),
                nn.Tanh(),
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._layers(input)
