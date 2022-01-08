import torch
import torch.nn as nn
import kornia


class DiffAugment(nn.Module):

    def __init__(self):
        super().__init__()

        self._layers = nn.Sequential(
                kornia.augmentation.ColorJitter(
                        p=1.0,
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.0,
                    ),
                kornia.augmentation.RandomHorizontalFlip(
                        p=0.5,
                    ),
                kornia.augmentation.RandomAffine(
                        p=0.5,
                        degrees=10,
                        scale=(0.7, 1.3),
                        translate=(0, 0),
                        shear=(0, 0),
                        padding_mode='zeros',
                    )
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self._layers(images)