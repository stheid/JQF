import numpy as np


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
