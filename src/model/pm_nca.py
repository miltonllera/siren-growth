import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import Callable, Literal
from jaxtyping import Float, Array

from src.nn.ca import run_ca
from src.nn.stochastic import DiagonalGaussian
from src.nn.modulated import ModulatedSIREN
from src.nn.perception import init_perception
from src.nn.update import init_update_fn, max_pool_alive, stochastic_update
from src.nn.seeding import mask_oval


class SIREN_NCA(eqx.Module):
    state_size: int
    encoder: Callable
    latent: DiagonalGaussian
    siren: Callable
    perception_fn: Callable
    update_fn: Callable
    update_prob: float
    alive_index: int | None
    alive_threshold: float
    num_dev_steps: tuple[int, int]

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        encoder_backbone: Callable,
        latent_size: int = 64,
        siren_width: int = 128,
        siren_depth: int = 3,
        pre_pattern_init: bool = True,
        nca_hidden_size: int = 12,
        perception_type: Literal[
            'sobel',
            'sobel-with-laplace',
            'learned',
            'laplace-wrapped',
            'laplace-same'
        ] = 'laplace-wrapped',
        update_width: int = 128,
        update_depth: int = 1,
        update_prob: float = 0.5,
        alive_index: int | None = 3,
        alive_threshold: float = 0.1,
        num_dev_steps = (10, 15),
        *,
        key
    ) -> None:
        super().__init__()

        C, H, W = input_shape
        state_size = nca_hidden_size + C
        cond_key, siren_key, update_key = jr.split(key, 3)

        encoder_output = encoder_backbone(jnp.zeros((C, H, W))).shape[0]
        latent = DiagonalGaussian(encoder_output, latent_size, cond_key)

        siren = ModulatedSIREN(
            2,
            output_size=C - (1 if alive_index is not None else 0),  # set init alive cells manually
            hidden_size=siren_width,
            depth=siren_depth,
            latent_size=latent_size,
            gain=30,
            modulation_type='scale',
            key=siren_key,
        )

        perception_fn = init_perception(perception_type, state_size, key)
        dummy_state = jnp.zeros((state_size, 8, 8))
        perception_out_size = perception_fn(dummy_state).shape[0]

        update_fn = init_update_fn(
            perception_out_size, update_width, update_depth, state_size, None, update_key
        )

        self.state_size = state_size
        self.encoder = encoder_backbone
        self.siren = siren
        self.latent = latent
        self.perception_fn = perception_fn
        self.update_fn = update_fn
        self.update_prob = update_prob
        self.alive_index = alive_index
        self.alive_threshold = alive_threshold
        self.num_dev_steps = num_dev_steps

    def __call__(
        self,
        input: Float[Array, "C H W"],
        key: jax.Array,
        steps=None,
        sample_latent=True,
        siren_noise_std: float = 0.0,
    ):
        C, H, W = input.shape
        z = self.latent(self.encoder(input), key, sample=False)
        x, y = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
        grid = jnp.stack([x, y], axis=-1).reshape(H * W, -1)
        init_state = jax.nn.sigmoid(
            self.siren(grid, z, key).reshape(H, W, -1).transpose(2, 0, 1)
        )
        if siren_noise_std > 0.0:
            key, noise_key = jr.split(key)
            init_state = init_state + siren_noise_std * jr.normal(noise_key, init_state.shape)

        if self.alive_index is not None:
            alive_mask = mask_oval((H, W), radius_x=0.25, radius_y=0.5)[None]
            # alive_mask = mask_oval((H, W), radius_x=0.16, radius_y=0.32)[None]
            init_state = jnp.concat([init_state * alive_mask, alive_mask])

        if self.state_size - C > 0:
            init_state = jnp.concat([init_state, jnp.zeros((self.state_size - C, H, W))])

        perception_fn = lambda x, s, key: self.perception_fn(x, key=key)

        def update_fn(cell_states, perception, key):
            update_mask = stochastic_update(cell_states.shape, self.update_prob, key)
            new_states = cell_states + self.update_fn(perception) * update_mask
            if self.alive_index is not None:
                pre_alive = max_pool_alive(cell_states, self.alive_index, self.alive_threshold)
                post_alive = max_pool_alive(new_states, self.alive_index, self.alive_threshold)
                new_states = new_states * pre_alive * post_alive
            return new_states

        x, dev_path = run_ca(
            init_state,
            perception_fn,
            update_fn,
            lambda x, *, key: 0.0,
            self.num_dev_steps if steps is None else steps,
            key
        )
        return x[:C], dev_path

    def __str__(self) -> str:
        #NOTE: very illegible but essentially, pic the class name from the output of `type`
        return str(type(self)).split(' ')[-1].split('.')[-1][:-2]
