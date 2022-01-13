import os
import zipfile
import urllib.request
from typing import Any, Dict

import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


_dataset = {
    'url': 'http://silentz.ml/few-shot-image-datasets.zip',
    'arc': 'archive.zip',
    'dst': 'few-shot-images',
}


def _ensure_dataset(root: str,
                    url: str,
                    archive: str,
                    dest: str):
    dst_path = os.path.join(root, dest)
    arc_path = os.path.join(root, archive)

    if not os.path.isdir(dst_path):
        urllib.request.urlretrieve(url, arc_path)

        if arc_path.endswith('.zip'):
            with zipfile.ZipFile(arc_path, 'r') as zip:
                zip.extractall(root)


class FewShotImageDatasetMixin:

    def __init__(self, root: str):
        _ensure_dataset(
                root=root,
                url=_dataset['url'],
                archive=_dataset['arc'],
                dest=_dataset['dst'],
            )


class FewShotImageDataset(FewShotImageDatasetMixin, Dataset):

    def __init__(self, root: str,
                       subdir: str):
        super().__init__(root)
        self._root = os.path.join(root, subdir)
        self._files = os.listdir(self._root)

        self._transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize((1024, 1024)),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
            ])

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = os.path.join(self._root, self._files[idx])
        image = Image.open(img_path).convert('RGB')
        image = torch.from_numpy(np.array(image))
        image = image.permute(2, 0, 1)
        image = self._transforms(image)
        return {'image': image}


class NoiseDataset(Dataset):

    def __init__(self, size: int, channels: int):
        self._size = size
        self._channels = channels

    def __len__(self):
        return self._size

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.zeros(self._channels, 1, 1).normal_(0.0, 1.0)
