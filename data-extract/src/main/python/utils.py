import os
import struct
from typing import List

import numpy as np
from more_itertools import flatten
from tqdm import tqdm


def remove_lsb(array: np.ndarray) -> np.ndarray:
    '''
    delete least sigificant bits from the bytes in the array, basically rounding the values down to powers of 2

    :param array: the array is expected to contain only integers that fit into one byte
    '''
    data = array.astype(np.uint8)
    mask = data > 0
    # calculate the power of base 2, round down and convert to int
    powers = np.log2(data[mask]).astype(np.uint8)

    # shifting ones to the appropriate bit
    data[mask] = np.ones_like(data[mask]) << powers
    return data


def load_jqf(dataset_path: str):
    """
    loads JQF dataset: used for testing transformer_algo.py
    """

    # prepare sequences
    seqs: List[bytes] = []
    with open(os.path.join(dataset_path, "events.bin"), "rb") as f:
        while length := f.read(4):
            n = struct.unpack(">i", length)[0]
            # seqs.append(" ".join(map(str, flatten(struct.iter_unpack(">h", f.read(n * 2))))))
            seqs.append(f.read(n * 2))

    # prepare files
    files: List[bytes] = []
    corpus = os.path.join(dataset_path, "corpus")
    for fname in tqdm(sorted(os.listdir(corpus))):
        with open(os.path.join(corpus, fname), "rb") as file:
            # files.append(" ".join(map(str, flatten(struct.iter_unpack(">B", file.read())))))
            files.append(file.read())

    return seqs, files
