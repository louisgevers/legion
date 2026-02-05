import numpy as np
from scipy.spatial.transform import Rotation


class NumpyBackend:
    name = "numpy"

    # array ops
    array = staticmethod(np.array)
    zeros = staticmethod(np.zeros)
    ones = staticmethod(np.ones)
    concatenate = staticmethod(np.concatenate)
    roll = staticmethod(np.roll)
    clip = staticmethod(np.clip)
    norm = staticmethod(np.linalg.norm)
    exp = staticmethod(np.exp)
    sin = staticmethod(np.sin)
    cos = staticmethod(np.cos)
    sum = staticmethod(np.sum)
    any = staticmethod(np.any)
    all = staticmethod(np.all)

    @staticmethod
    def quat2euler(xyzw):
        return Rotation.from_quat(xyzw).as_euler("xyz", degrees=False)

    @staticmethod
    def quat_rotate(xyzw, v, inverse: bool = False):
        return Rotation.from_quat(xyzw).apply(v, inverse=inverse)

    # execution transforms
    @staticmethod
    def jit(fn, **kwargs):
        return fn  # no-op

    # RNG
    @staticmethod
    def rng_seed(seed: int):
        # Numpy RNG are stateful, construct generator once per seed (expensive otherwise)
        return np.random.default_rng(seed)

    @staticmethod
    def rng_split(key, num: int = 2):
        # Fake split, the generator is stateful and expensive to instantiate each time
        return (key,) * num

    @staticmethod
    def rng_uniform(key, shape, minval=0.0, maxval=1.0):
        return key.uniform(minval, maxval, size=shape)

    @staticmethod
    def rng_normal(key, shape, mean=0.0, std=1.0):
        return key.normal(mean, std, size=shape)
