import os
import zipfile
import urllib.request
from torch.utils.data import Dataset
from typing import Any, Dict
import torchvision


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

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = os.path.join(self._root, self._files[idx])
        image = torchvision.io.read_image(img_path, torchvision.io.ImageReadMode.RGB)
        return {'image': image}

