import torch
import torch.nn as nn
from enum import Enum

class ImageType(Enum):
    REAL_UP_L = 0
    REAL_UP_R = 1
    REAL_DOWN_R = 2
    REAL_DOWN_L = 3
    FAKE = 4


def crop_image_part(image: torch.Tensor,
                    part: ImageType) -> torch.Tensor:
    size = image.shape[2] // 2

    if part == ImageType.REAL_UP_L:
        return image[:, :, :size, :size]

    elif part == ImageType.REAL_UP_R:
        return image[:, :, :size, size:]

    elif part == ImageType.REAL_DOWN_L:
        return image[:, :, size:, :size]

    elif part == ImageType.REAL_DOWN_R:
        return image[:, :, size:, size:]

    else:
        raise ValueError('invalid part')


def init_weights(module: nn.Module):
    if isinstance(module, nn.Conv2d):
        torch.nn.init.normal_(module.weight, 0.0, 0.02)

    if isinstance(module, nn.BatchNorm2d):
        torch.nn.init.normal_(module.weight, 1.0, 0.02)
        module.bias.data.fill_(0)
