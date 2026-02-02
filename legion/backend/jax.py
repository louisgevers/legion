import jax.numpy as jnp


class JaxBackend:
    name = "jax"

    array = staticmethod(jnp.array)
    zeros = staticmethod(jnp.zeros)
    ones = staticmethod(jnp.ones)
    concatenate = staticmethod(jnp.concatenate)
    clip = staticmethod(jnp.clip)
    norm = staticmethod(jnp.linalg.norm)
    sin = staticmethod(jnp.sin)
    cos = staticmethod(jnp.cos)
