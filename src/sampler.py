from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
import numpy as np


class InfiniteSampler(Sampler):

    def __init__(self, source: Dataset):
        self._length = len(source)

    def __len__(self):
        return 2 ** 31

    def __iter__(self):
        def iter_func():
            permutation = np.random.permutation(self._length)
            idx = 0

            while True:
                yield permutation[idx]
                idx += 1

                if idx >= self._length:
                    permutation = np.random.permutation(self._length)
                    idx = 0

        return iter(iter_func())

