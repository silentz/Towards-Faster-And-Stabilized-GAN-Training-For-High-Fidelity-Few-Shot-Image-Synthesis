import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum
from typing import Any, Tuple, Union

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

    def forward(self, input: torch.Tensor) -> \
            torch.Tensor:
            #  Tuple[torch.Tensor, torch.Tensor]:
        size_4  = self._init(input)
        size_8  = self._upsample_8(size_4)
        size_16 = self._upsample_16(size_8)
        size_32 = self._upsample_32(size_16)

        size_64  = self._sle_64 (size_4,  self._upsample_64 (size_32) )
        size_128 = self._sle_128(size_8,  self._upsample_128(size_64) )
        size_256 = self._sle_256(size_16, self._upsample_256(size_128))
        size_512 = self._sle_512(size_32, self._upsample_512(size_256))

        size_1024 = self._upsample_1024(size_512)

        #  out_128  = self._out_128 (size_128)
        out_1024 = self._out_1024(size_1024)

        #  return out_128, out_1024
        return out_1024


class Discriminrator(nn.Module):

    class ImageType(Enum):
        REAL_UP_L = 0
        REAL_UP_R = 1
        REAL_DOWN_R = 2
        REAL_DOWN_L = 3
        FAKE = 4

    def __init__(self, in_channels: int):
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

        self._init = nn.Sequential(
                SpectralConv2d(
                        in_channels=in_channels,
                        out_channels=self._channels[1024],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                nn.LeakyReLU(negative_slope=0.2),
                SpectralConv2d(
                        in_channels=self._channels[1024],
                        out_channels=self._channels[512],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                nn.BatchNorm2d(num_features=self._channels[512]),
                nn.LeakyReLU(negative_slope=0.2),
            )

        self._downsample_256 = DownsampleBlockT2(in_channels=self._channels[512], out_channels=self._channels[256])
        self._downsample_128 = DownsampleBlockT2(in_channels=self._channels[256], out_channels=self._channels[128])
        self._downsample_64  = DownsampleBlockT2(in_channels=self._channels[128], out_channels=self._channels[64] )
        self._downsample_32  = DownsampleBlockT2(in_channels=self._channels[64],  out_channels=self._channels[32] )
        self._downsample_16  = DownsampleBlockT2(in_channels=self._channels[32],  out_channels=self._channels[16] )

        self._sle_64 = SLEBlock(in_channels=self._channels[512], out_channels=self._channels[64])
        self._sle_32 = SLEBlock(in_channels=self._channels[256], out_channels=self._channels[32])
        self._sle_16 = SLEBlock(in_channels=self._channels[128], out_channels=self._channels[16])

        self._small_track = nn.Sequential(
                SpectralConv2d(
                        in_channels=in_channels,
                        out_channels=self._channels[256],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                nn.LeakyReLU(negative_slope=0.2),
                DownsampleBlockT1(in_channels=self._channels[256], out_channels=self._channels[128]),
                DownsampleBlockT1(in_channels=self._channels[128], out_channels=self._channels[64] ),
                DownsampleBlockT1(in_channels=self._channels[64],  out_channels=self._channels[32] ),
            )

        self._features_large = nn.Sequential(
                SpectralConv2d(
                        in_channels=self._channels[16] ,
                        out_channels=self._channels[8],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                nn.BatchNorm2d(num_features=self._channels[8]),
                nn.LeakyReLU(negative_slope=0.2),
                SpectralConv2d(
                        in_channels=self._channels[8],
                        out_channels=1,
                        kernel_size=4,
                        stride=1,
                        padding=0,
                        bias=False,
                    )
            )

        self._features_small = nn.Sequential(
                SpectralConv2d(
                        in_channels=self._channels[32],
                        out_channels=1,
                        kernel_size=4,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
            )

        self._decoder_large = Decoder(in_channels=self._channels[16], out_channels=3)
        self._decoder_small = Decoder(in_channels=self._channels[32], out_channels=3)
        self._decoder_piece = Decoder(in_channels=self._channels[32], out_channels=3)

    def forward(self, images: torch.Tensor, image_type: 'Discriminrator.ImageType') -> \
            Union[
                torch.Tensor,
                Tuple[torch.Tensor, Tuple[Any, Any, Any]]
            ]:
        # large track

        down_512 = self._init(images)
        down_256 = self._downsample_256(down_512)
        down_128 = self._downsample_128(down_256)

        down_64 = self._downsample_64(down_128)
        down_64 = self._sle_64(down_512, down_64)

        down_32 = self._downsample_32(down_64)
        down_32 = self._sle_32(down_256, down_32)

        down_16 = self._downsample_16(down_32)
        down_16 = self._sle_16(down_128, down_16)

        # small track

        small_images = F.interpolate(images, size=128)
        down_small = self._small_track(small_images)

        # features

        features_large = self._features_large(down_16).view(-1)
        features_small = self._features_small(down_small).view(-1)
        features = torch.cat([features_large, features_small], dim=0)

        # decoder

        if image_type != Discriminrator.ImageType.FAKE:
            dec_large = self._decoder_large(down_16)
            dec_small = self._decoder_small(down_small)
            dec_piece = None

            if image_type == Discriminrator.ImageType.REAL_UP_L:
                dec_piece = self._decoder_piece(down_32[:, :, :8, :8])

            if image_type == Discriminrator.ImageType.REAL_UP_R:
                dec_piece = self._decoder_piece(down_32[:, :, :8, 8:])

            if image_type == Discriminrator.ImageType.REAL_DOWN_L:
                dec_piece = self._decoder_piece(down_32[:, :, 8:, :8])

            if image_type == Discriminrator.ImageType.REAL_DOWN_R:
                dec_piece = self._decoder_piece(down_32[:, :, 8:, 8:])

            return features, (dec_large, dec_small, dec_piece)

        return features
