import torch
import torch.nn as nn
import kornia


class DiffAugment(nn.Module):

    def __init__(self):
        super().__init__()

        self._layers = nn.Sequential(
                kornia.augmentation.ColorJitter(
                        p=1.0,
                        brightness=0.5,
                        contrast=0.5,
                        saturation=0.5,
                        hue=0.0,
                    ),
                kornia.augmentation.RandomAffine(
                        p=0.5,
                        degrees=0,
                        scale=(0.7, 1.3),
                        translate=(0.125, 0.125),
                        shear=(0, 0),
                        padding_mode='zeros',
                    ),
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self._layers(images)
