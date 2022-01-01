import torch
from dataclasses import dataclass


@dataclass
class Batch:
    images: torch.Tensor

    def to(self, device: torch.device) -> 'Batch':
        return Batch(
                images=self.images.to(device),
            )


def collate_fn(inputs: list) -> Batch:
    images = [x['image'] for x in inputs]
    images = torch.stack(images, dim=0)

    return Batch(
            images=images,
        )
