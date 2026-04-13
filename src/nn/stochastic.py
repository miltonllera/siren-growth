import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn


class DiagonalGaussian(eqx.Module):
    linear: nn.Linear

    def __init__(self, input_size, latent_size, key):
        super().__init__()
        self.linear = nn.Linear(input_size, 2 * latent_size, key=key)

    def __call__(self, h, key, sample=True, return_parmas=False):
        mean, log_std = jnp.split(self.linear(h), 2)
        if sample:
            eps = jr.normal(key, (len(h),))
            z = mean + jnp.exp(0.5 * log_std) * eps
        else:
            z = mean
        return z if not return_parmas else (z, (mean, log_std))
