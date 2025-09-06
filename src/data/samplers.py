import random
from typing import List, Iterator

import pandas as pd
from torch.utils.data import Sampler, Dataset

class RandomSlidingWindowSampler(Sampler[List[int]]):

    __slots__ = ["window_size", "_dataset", "_indices"]

    def __init__(self, dataset: Dataset, window_size: int, seed: int = 42) -> None:
        super().__init__()
        self.window_size = window_size
        self._dataset = dataset
        self._indices: List[int] = list(range(len(dataset) - window_size + 1))
        random.seed(seed)

    def __iter__(self) -> Iterator[List[int]]:
        return (
            list(range(start, start + self.window_size))
            for start in random.sample(self._indices, len(self._indices))
        )

    def __len__(self) -> int:
        return len(self._indices)