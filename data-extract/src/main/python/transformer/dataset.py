import logging
import numpy as np
import struct
from more_itertools import flatten
from typing import Tuple

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Dataset:
    def __init__(self, X: np.array = None, y: np.array = None,
                 max_size=10000, new_sw=2, weights=None, bitsize=8):
        self.bitsize = bitsize  # bytes -> 8, short -> 16, int -> 32

        if X is None and y is None:
            self.X = np.array([])
            self.y = np.array([])
        else:
            if bitsize == 16:
                X = flatten(struct.iter_unpack('>H', X))
            elif bitsize == 32:
                X = flatten(struct.iter_unpack('>I', X))
            else:
                X = flatten(struct.iter_unpack('>B', X))
            if (type(X) != np.ndarray) or (type(y) != np.ndarray):
                # todo: convert to list[lists] not np.
                self.X = np.array(X)
                self.y = np.array(y)
                if len(self.X.shape) != 1 and len(self.y.shape) != 1:
                    raise RuntimeError(
                        f"X and y should be always 1 dimensional, but found: X.shape: {self.X.shape}, y.shape: "
                        f"{self.y.shape}")
            else:
                self.X = X
                self.y = y

        logger.debug(f'Dataset:init(X,y) -> {len(self.X)}, {len(self.y)}')
        self.max_size = max_size
        self.new_sw = new_sw  # sample weights
        if weights is None and X is not None:
            self.weights = np.ones(len(self))
        else:
            self.weights = np.array([])

    def __getitem__(self, value: Tuple):
        return Dataset(self.X[value], self.y[value], max_size=self.max_size,
                       new_sw=self.new_sw, weights=None)

    def __iadd__(self, other):
        for attr in ['X', 'y']:
            old = getattr(self, attr)
            new = getattr(other, attr)
            setattr(self, attr, np.append(old, new))

        # add old and new weights
        old = np.ones_like(self.weights)
        new = np.full(other.weights.shape, self.new_sw)
        self.weights = np.concatenate((old, new))
        return self

    def __iter__(self):
        yield self.X
        yield self.y

    def __len__(self):
        xlen = self.X.shape[0]
        ylen = self.y.shape[0]
        if xlen != ylen:
            raise RuntimeError('Label and Data is not of the same length. Dataset is borked')
        return xlen

    @property
    def is_empty(self):
        return len(self) == 0

    @property
    def xdim(self):
        return self.X.shape[0]

    @property
    def ydim(self):
        return self.y.shape[0]

    def split(self, frac=.8):
        if self.is_empty:
            raise RuntimeError('Can\'t split an empty dataset')

        # returns split to train and validation datasets with given fraction
        x, y = [np.split(k, [int(frac * len(k))]) for k in self]
        return Dataset(x[0], y[0], max_size=self.max_size,
                       new_sw=self.new_sw, weights=None), Dataset(x[1], y[1], max_size=self.max_size,
                                                                  new_sw=self.new_sw, weights=None)
