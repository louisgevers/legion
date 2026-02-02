def get_backend(name: str):
    # Lazy import backends
    if name == "numpy":
        from .numpy import NumpyBackend

        return NumpyBackend()

    if name == "jax":
        from .jax import JaxBackend

        return JaxBackend()

    raise ValueError(f"Unknown backend: {name}")
