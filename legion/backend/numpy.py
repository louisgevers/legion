import numpy as np


class NumpyBackend:
    name = "numpy"

    # array ops
    array = staticmethod(np.array)
    zeros = staticmethod(np.zeros)
    ones = staticmethod(np.ones)
    concatenate = staticmethod(np.concatenate)
    clip = staticmethod(np.clip)
    norm = staticmethod(np.linalg.norm)
    exp = staticmethod(np.exp)
    sin = staticmethod(np.sin)
    cos = staticmethod(np.cos)
    sum = staticmethod(np.sum)
    any = staticmethod(np.any)
    all = staticmethod(np.all)

    # execution transforms
    @staticmethod
    def jit(fn, **kwargs):
        return fn  # no-op
