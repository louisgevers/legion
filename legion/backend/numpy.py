import numpy as np


class NumpyBackend:
    name = "numpy"

    array = staticmethod(np.array)
    zeros = staticmethod(np.zeros)
    ones = staticmethod(np.ones)
    concatenate = staticmethod(np.concatenate)
    clip = staticmethod(np.clip)
    norm = staticmethod(np.linalg.norm)
    sin = staticmethod(np.sin)
    cos = staticmethod(np.cos)
