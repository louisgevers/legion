import jax
import jax.numpy as jnp


class JaxBackend:
    name = "jax"

    # array ops
    array = staticmethod(jnp.array)
    zeros = staticmethod(jnp.zeros)
    ones = staticmethod(jnp.ones)
    concatenate = staticmethod(jnp.concatenate)
    clip = staticmethod(jnp.clip)
    norm = staticmethod(jnp.linalg.norm)
    exp = staticmethod(jnp.exp)
    sin = staticmethod(jnp.sin)
    cos = staticmethod(jnp.cos)
    sum = staticmethod(jnp.sum)
    any = staticmethod(jnp.any)
    all = staticmethod(jnp.all)

    # execution transforms
    @staticmethod
    def jit(fn, **kwargs):
        return jax.jit(fn, **kwargs)

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
