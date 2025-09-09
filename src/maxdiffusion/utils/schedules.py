import jax.numpy as jnp


def cosine_fade(T: int, start: float = 1.0, end: float = 0.0):
    """Cosine curve that fades from ``start`` to ``end`` over ``T`` steps."""
    t = jnp.linspace(0.0, 1.0, T)
    return end + (start - end) * 0.5 * (1.0 + jnp.cos(jnp.pi * t))


def linear_ramp(T: int, start: float = 0.0, end: float = 3.0):
    """Linear ramp from ``start`` to ``end`` over ``T`` steps."""
    t = jnp.linspace(0.0, 1.0, T)
    return start + t * (end - start)
