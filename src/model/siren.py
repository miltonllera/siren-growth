import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import Callable
from jaxtyping import Float, Array

from src.nn.stochastic import DiagonalGaussian
from src.nn.modulated import ModulatedSIREN


class SIREN(eqx.Module):
    encoder: Callable
    latent: DiagonalGaussian
    siren: Callable

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        encoder_backbone: Callable,
        latent_size: int = 64,
        siren_width: int = 128,
        siren_depth: int = 3,
        *,
        key
    ) -> None:
        super().__init__()
        C, H, W = input_shape
        cond_key, siren_key = jr.split(key, 2)

        encoder_output = encoder_backbone(jnp.zeros((C, H, W))).shape[0]
        latent = DiagonalGaussian(encoder_output, latent_size, cond_key)
        siren = ModulatedSIREN(
            input_size=2,
            output_size=C,
            hidden_size=siren_width,
            depth=siren_depth,
            latent_size=latent_size,
            gain=30,
            modulation_type='scale',
            key=siren_key,
        )

        self.encoder = encoder_backbone
        self.siren = siren
        self.latent = latent

    def __call__(self, input: Float[Array, "C H W"], key: jax.Array, steps=None, sample_latent=True):
        _, H, W = input.shape
        z, params = self.latent(self.encoder(input), key, sample=False, return_parmas=True)
        x, y = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
        grid = jnp.stack([x, y], axis=-1).reshape(H * W, -1)
        recons = self.siren(grid, z, key).reshape(H, W, -1).transpose(2, 0, 1)
        return recons, z, params


    def __str__(self) -> str:
        #NOTE: very illegible but essentially, pic the class name from the output of `type`
        return str(type(self)).split(' ')[-1].split('.')[-1][:-2]

