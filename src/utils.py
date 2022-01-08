import torch
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
