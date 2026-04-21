# SPDX-FileCopyrightText: Copyright (c) 2026 Louis Gevers
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE file for full license information.
#
import numpy as np
from scipy.spatial.transform import Rotation


class NumpyBackend:
    name = "numpy"

    # constants
    pi = np.pi

    # array ops
    array = staticmethod(np.array)
    zeros = staticmethod(np.zeros)
    ones = staticmethod(np.ones)
    arange = staticmethod(np.arange)
    concatenate = staticmethod(np.concatenate)
    append = staticmethod(np.append)
    tile = staticmethod(np.tile)
    where = staticmethod(np.where)
    roll = staticmethod(np.roll)
    clip = staticmethod(np.clip)
    norm = staticmethod(np.linalg.norm)
    exp = staticmethod(np.exp)
    sin = staticmethod(np.sin)
    cos = staticmethod(np.cos)
    tan = staticmethod(np.tan)
    atan2 = staticmethod(np.atan2)
    sum = staticmethod(np.sum)
    square = staticmethod(np.square)
    sqrt = staticmethod(np.sqrt)
    abs = staticmethod(np.abs)
    any = staticmethod(np.any)
    all = staticmethod(np.all)
    isnan = staticmethod(np.isnan)

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

    @staticmethod
    def scan(fn, init, xs, length):
        if xs is None:
            xs = [None] * length
        carry = init
        ys = []
        for x in xs:
            carry, y = fn(carry, x)
            ys.append(y)
        return carry, np.stack(ys)

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
    def rng_exponential(key, shape):
        return key.uniform(size=shape)

    @staticmethod
    def rng_bernoulli(key, p, shape):
        return key.binomial(n=1, p=p, size=shape)

    @staticmethod
    def rng_normal(key, shape, mean=0.0, std=1.0):
        return key.normal(mean, std, size=shape)
