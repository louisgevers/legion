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
