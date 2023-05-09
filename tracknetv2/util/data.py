import numpy as np

from itertools import chain
from typing import Union, List


class SparseContainer:
    def __init__(self, data: Union[List[Union[List, np.array]], np.array]):
        self.spans = [len(x) for x in data]
        self.starts = np.cumsum([0] + self.spans[:-1])
        self.spans = np.array(self.spans)
        self.length = len(data)
        if isinstance(data, list):
            self.data = np.array(list(chain.from_iterable(data)))
        elif isinstance(data, np.array):
            self.data = data.reshape(-1, )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError(f"Index {idx} is out of bounds.")
        start_pos = self.starts[idx]
        return self.data[start_pos: start_pos + self.spans[idx]]
