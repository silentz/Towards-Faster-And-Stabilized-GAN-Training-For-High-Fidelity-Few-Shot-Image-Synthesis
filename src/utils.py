import torch
import torch.nn as nn
from .models import Discriminrator


def crop_image_part(image: torch.Tensor,
                    part: Discriminrator.ImageType) -> torch.Tensor:
    size = image.shape[2] // 2

    if part == Discriminrator.ImageType.REAL_UP_L:
        return image[:, :, :size, :size]

    elif part == Discriminrator.ImageType.REAL_UP_R:
        return image[:, :, :size, size:]

    elif part == Discriminrator.ImageType.REAL_DOWN_L:
        return image[:, :, size:, :size]

    elif part == Discriminrator.ImageType.REAL_DOWN_R:
        return image[:, :, size:, size:]

    else:
        raise ValueError('invalid part')


def init_weights(module: nn.Module):
    if isinstance(module, nn.Conv2d):
        torch.nn.init.normal(module.weight, 0.0, 0.02)

    if isinstance(module, nn.BatchNorm2d):
        torch.nn.init.normal(module.weight, 1.0, 0.02)
        module.bias.data.fill_(0)
