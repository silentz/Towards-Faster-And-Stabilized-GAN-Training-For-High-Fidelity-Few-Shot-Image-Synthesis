import torch
import torch.nn as nn
from typing import Tuple

from .layers import (
    SpectralConv2d,
    InitLayer,
    SLEBlock,
    UpsampleBlockT1,
    UpsampleBlockT2,
    DownsampleBlockT1,
    DownsampleBlockT2,
    Decoder,
)


class Generator(nn.Module):

    def __init__(self, in_channels: int,
                       out_channels: int):
        super().__init__()

        self._channels = {
                4:    1024,
                8:    512,
                16:   256,
                32:   128,
                64:   128,
                128:  64,
                256:  32,
                512:  16,
                1024: 8,
            }

        self._init = InitLayer(
                in_channels=in_channels,
                out_channels=self._channels[4],
            )

        self._upsample_8    = UpsampleBlockT2(in_channels=self._channels[4],   out_channels=self._channels[8]   )
        self._upsample_16   = UpsampleBlockT1(in_channels=self._channels[8],   out_channels=self._channels[16]  )
        self._upsample_32   = UpsampleBlockT2(in_channels=self._channels[16],  out_channels=self._channels[32]  )
        self._upsample_64   = UpsampleBlockT1(in_channels=self._channels[32],  out_channels=self._channels[64]  )
        self._upsample_128  = UpsampleBlockT2(in_channels=self._channels[64],  out_channels=self._channels[128] )
        self._upsample_256  = UpsampleBlockT1(in_channels=self._channels[128], out_channels=self._channels[256] )
        self._upsample_512  = UpsampleBlockT2(in_channels=self._channels[256], out_channels=self._channels[512] )
        self._upsample_1024 = UpsampleBlockT1(in_channels=self._channels[512], out_channels=self._channels[1024])

        self._sle_64  = SLEBlock(in_channels=self._channels[4],  out_channels=self._channels[64] )
        self._sle_128 = SLEBlock(in_channels=self._channels[8],  out_channels=self._channels[128])
        self._sle_256 = SLEBlock(in_channels=self._channels[16], out_channels=self._channels[256])
        self._sle_512 = SLEBlock(in_channels=self._channels[32], out_channels=self._channels[512])

        self._out_128 = nn.Sequential(
                SpectralConv2d(
                    in_channels=self._channels[128],
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding='same',
                    bias=False,
                ),
                nn.Tanh(),
            )

        self._out_1024 = nn.Sequential(
                SpectralConv2d(
                    in_channels=self._channels[1024],
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding='same',
                    bias=False,
                ),
                nn.Tanh(),
            )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        size_4  = self._init(input)
        size_8  = self._upsample_8(size_4)
        size_16 = self._upsample_16(size_8)
        size_32 = self._upsample_32(size_16)


        size_64  = self._sle_64 (size_4,  self._upsample_64 (size_32) )
        size_128 = self._sle_128(size_8,  self._upsample_128(size_64) )
        size_256 = self._sle_256(size_16, self._upsample_256(size_128))
        size_512 = self._sle_512(size_32, self._upsample_512(size_256))

        size_1024 = self._upsample_1024(size_512)

        out_128  = self._out_128 (size_128)
        out_1024 = self._out_1024(size_1024)

        return out_128, out_1024
