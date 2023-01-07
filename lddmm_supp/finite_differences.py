



import jax.numpy as jnp


SPACING = 1


def spatial_gradient(arr):
    dx, dy =   jnp.gradient(arr)
    return [SPACING*dx, SPACING*dy]

