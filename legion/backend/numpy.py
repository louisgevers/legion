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

    # RNG
    @staticmethod
    def rng_seed(seed: int):
        return np.random.SeedSequence(seed)

    @staticmethod
    def rng_split(key, num: int = 2):
        return tuple(key.spawn(num))

    @staticmethod
    def rng_uniform(key, shape, minval=0.0, maxval=1.0):
        return np.random.default_rng(key).uniform(minval, maxval, size=shape)

    @staticmethod
    def rng_normal(key, shape, mean=0.0, std=1.0):
        return np.random.default_rng(key).normal(mean, std, size=shape)
