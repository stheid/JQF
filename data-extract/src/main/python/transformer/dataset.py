import logging
import random
import struct
from typing import Tuple

import numpy as np
from more_itertools import flatten

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Dataset:
    def __init__(self, X=None, y=None, max_size=10000, new_sw=2, weights=None, bitsize=8):
        self.bitsize = bitsize  # bytes -> 8, short -> 16, int -> 32

        if X is None and y is None:
            self.X = []
            self.y = []
        else:
            if type(X[0]) == bytes:
                if bitsize == 16:
                    X = [list(flatten(struct.iter_unpack('>H', x))) for x in X]
                elif bitsize == 32:
                    X = [list(flatten(struct.iter_unpack('>I', x))) for x in X]
                else:
                    X = [list(flatten(struct.iter_unpack('>B', x))) for x in X]

            self.X = X
            self.y = y

        logger.debug(f'Dataset:init(X,y) -> {len(self.X)}, {len(self.y)}')
        self.max_size = max_size
        self.new_sw = new_sw  # sample weights
        # todo: change weights to list from numpy
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
            old.extend(new)
            setattr(self, attr, old)

        # todo: change weights to list from numpy
        # add old and new weights
        old = np.ones_like(self.weights)
        new = np.full(other.weights.shape, self.new_sw)
        self.weights = np.concatenate((old, new))
        return self

    def sample(self, n):
        if len(self) < n:
            # w/ replacement
            return random.choices(self.X, k=n)
        else:
            # wo/ replacement
            return random.sample(self.X, k=n)

    def __iter__(self):
        yield self.X
        yield self.y

    def __len__(self):
        return len(self.X)

    @property
    def is_empty(self):
        return len(self) == 0

    @property
    def xdim(self):
        return len(self.X)

    @property
    def ydim(self):
        return len(self.y)

    def split(self, frac=.8, shuffled=True):
        if self.is_empty:
            raise RuntimeError('Can\'t split an empty dataset')

        combined_list = list(zip(*self))

        # Shuffle the list of tuples
        if shuffled:
            random.shuffle(combined_list)

        # returns split to train and validation datasets with given fraction
        x, y = [[list(k[:int(frac * len(k))]), list(k[int(frac * len(k)):])] for k in zip(*combined_list)]
        return Dataset(x[0], y[0], max_size=self.max_size,
                       new_sw=self.new_sw, weights=None, bitsize=self.bitsize), \
            Dataset(x[1] or None, y[1] or None, max_size=self.max_size, new_sw=self.new_sw, weights=None,
                    bitsize=self.bitsize)
