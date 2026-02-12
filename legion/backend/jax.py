import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation


class JaxBackend:
    name = "jax"

    def __init__(self):
        # Required for Ampere architectures (e.g., RTX30 and RTX40) or NaNs start showing up
        jax.config.update("jax_default_matmul_precision", "highest")

    # array ops
    array = staticmethod(jnp.array)
    zeros = staticmethod(jnp.zeros)
    ones = staticmethod(jnp.ones)
    concatenate = staticmethod(jnp.concatenate)
    where = staticmethod(jnp.where)
    roll = staticmethod(jnp.roll)
    clip = staticmethod(jnp.clip)
    norm = staticmethod(jnp.linalg.norm)
    exp = staticmethod(jnp.exp)
    sin = staticmethod(jnp.sin)
    cos = staticmethod(jnp.cos)
    sum = staticmethod(jnp.sum)
    square = staticmethod(jnp.square)
    any = staticmethod(jnp.any)
    all = staticmethod(jnp.all)
    isnan = staticmethod(jnp.isnan)

    @staticmethod
    def quat2euler(xyzw):
        return Rotation.from_quat(xyzw).as_euler("xyz", degrees=False)

    @staticmethod
    def quat_rotate(xyzw, v, inverse: bool = False):
        return Rotation.from_quat(xyzw).apply(v, inverse=inverse)

    # execution transforms
    @staticmethod
    def jit(fn, **kwargs):
        return jax.jit(fn, **kwargs)

    @staticmethod
    def scan(fn, init, xs, length):
        return jax.lax.scan(fn, init, xs, length)

    # RNG
    @staticmethod
    def rng_seed(seed: int):
        return jax.random.PRNGKey(seed)

    @staticmethod
    def rng_split(key, num: int = 2):
        return jax.random.split(key, num)

    @staticmethod
    def rng_uniform(key, shape, minval=0.0, maxval=1.0):
        return jax.random.uniform(key, shape, minval=minval, maxval=maxval)

    @staticmethod
    def rng_normal(key, shape, mean=0.0, std=1.0):
        return mean + std * jax.random.normal(key, shape)
