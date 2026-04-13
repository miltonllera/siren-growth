import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import Callable, Literal
from jaxtyping import Float, Array

from src.nn.ca import run_ca
from src.nn.perception import init_perception
from src.nn.morphogens import init_morphogen_fn
from src.nn.update import init_update_fn, max_pool_alive, stochastic_update
from src.nn.stochastic import DiagonalGaussian
# from src.nn.conditioning import LinearConditioning


class NCA(eqx.Module):
    state_size: int
    encoder: Callable
    latent: DiagonalGaussian
    latent_proj: Callable
    perception_fn: Callable
    update_fn: Callable
    update_prob: float
    conditioning_mode: str
    alive_index: int | None
    alive_threshold: float
    num_dev_steps: tuple[int, int]

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        hidden_size,
        encoder_backbone: Callable,
        latent_size: int,
        perception_type: Literal['sobel', 'sobel-with-laplace', 'learned'] = 'sobel-with-laplace',
        morphogen_type: Literal[
            'gaussian',
            'directional',
            'sinusoidal',
            'mixed',
        ] | None = None,
        update_width: int = 128,
        update_depth: int = 1,
        update_prob: float = 0.5,
        alive_index: int | None = 3,
        alive_threshold: float = 0.1,
        conditioning_mode: Literal['additive', 'concat'] = "additive",
        num_dev_steps = (48, 96),
        *,
        key
    ) -> None:
        super().__init__()
        C, H, W = input_shape
        assert conditioning_mode == 'additive' or conditioning_mode == 'concat'

        state_size = hidden_size + 4
        conv_key, update_key, cond_key = jr.split(key, 3)

        encoder_output = encoder_backbone(jnp.zeros((C, H, W))).shape[0]
        latent = DiagonalGaussian(encoder_output, latent_size, cond_key)
        if latent_size != state_size:
            latent_proj = eqx.nn.Linear(latent_size, state_size, key=cond_key)
        else:
            latent_proj = lambda x: x

        perception_fn = init_perception(perception_type, state_size, key)
        dummy_state = jnp.zeros((state_size, 8, 8))
        perception_out_size = perception_fn(dummy_state, key=conv_key).shape[0]

        morphogen_fn = init_morphogen_fn(morphogen_type)
        morphogen_size = morphogen_fn(8, 8).shape[0]
        morphogen_concat = eqx.nn.Lambda(lambda x: jnp.concat([x, morphogen_fn(*x.shape[1:])]))

        layer_input_size = (
            perception_out_size + morphogen_size + state_size * (conditioning_mode == 'concat')
        )

        update_fn = init_update_fn(
            layer_input_size, update_width, update_depth, state_size, morphogen_concat, update_key
        )

        self.state_size = state_size
        self.perception_fn = perception_fn
        self.update_fn = update_fn
        self.update_prob = update_prob
        self.latent = latent
        self.latent_proj = latent_proj
        self.encoder = encoder_backbone
        self.conditioning_mode = conditioning_mode
        self.alive_index = alive_index
        self.alive_threshold = alive_threshold
        self.num_dev_steps = num_dev_steps

    def __call__(self, inputs: Float[Array, "C H W"], key: jax.Array, steps=None):
        goal = self.latent_proj(self.latent(self.encoder(inputs), key, sample=False))

        def perception_fn(cell_states, conditioning_vector, key):
            if self.conditioning_mode == 'additive':
                perception = self.perception_fn(cell_states + conditioning_vector, key=key)
            else:
                perception = self.perception_fn(cell_states, key=key)
                perception = jnp.concat([perception, conditioning_vector])
            return perception

        def update_fn(cell_states, perception, key):
            update_mask = stochastic_update(cell_states.shape, self.update_prob, key)
            new_states = cell_states + self.update_fn(perception) * update_mask
            if self.alive_index is not None:
                pre_alive = max_pool_alive(cell_states, self.alive_index, self.alive_threshold)
                post_alive = max_pool_alive(new_states, self.alive_index, self.alive_threshold)
                new_states = new_states * pre_alive * post_alive
            return new_states

        state_shape = self.state_size, *inputs.shape[1:]
        # init_state = jnp.ones_like(inputs, shape=state_shape).at[:4].set(0.0)
        init_state = jr.uniform(key, state_shape, minval=-0.1, maxval=0.1).at[:4].set(0.0)

        x, dev_path = run_ca(
            init_state,
            perception_fn,
            update_fn,
            lambda x, *, key: jnp.tile(goal[:, None, None], (1, *inputs.shape[1:])),
            self.num_dev_steps, key
        )
        return x[:4], dev_path

    def __str__(self) -> str:
        #NOTE: very illegible but essentially, pic the class name from the output of `type`
        return str(type(self)).split(' ')[-1].split('.')[-1][:-2]


class NoisePatternedNCA(eqx.Module):
    input_size: tuple[int, int]
    init_prepatterns: np.ndarray
    state_size: int
    perception_fn: Callable
    update_fn: Callable
    update_prob: float
    alive_index: int | None
    alive_threshold: float
    num_dev_steps: tuple[int, int]

    def __init__(
        self,
        n_targets:int,
        input_size: tuple[int, int],
        hidden_size:int,
        perception_type: Literal['sobel', 'sobel-with-laplace', 'learned'] = 'sobel-with-laplace',
        update_width: int = 128,
        update_depth: int = 1,
        update_prob: float = 0.5,
        alive_index: int | None = 3,
        alive_threshold: float = 0.1,
        num_dev_steps = (48, 96),
        *,
        key
    ) -> None:
        super().__init__()
        state_size = hidden_size + 4
        conv_key, update_key = jr.split(key)

        init_prepatterns = np.asarray(jr.normal(key, (n_targets, 4, *input_size)))

        perception_fn = init_perception(perception_type, state_size, key)
        dummy_state = jnp.zeros((state_size, 8, 8))
        perception_out_size = perception_fn(dummy_state, key=conv_key).shape[0]

        update_fn = init_update_fn(
            perception_out_size, update_width, update_depth, state_size, None, update_key
        )

        self.input_size = input_size
        self.state_size = state_size
        self.init_prepatterns = init_prepatterns
        self.perception_fn = perception_fn
        self.update_fn = update_fn
        self.update_prob = update_prob
        self.alive_index = alive_index
        self.alive_threshold = alive_threshold
        self.num_dev_steps = num_dev_steps

    def __call__(self, input_idx, key: jax.Array, steps=None):
        def perception_fn(cell_states, conditioning_vector, key):
            return self.perception_fn(cell_states, key=key)

        def update_fn(cell_states, perception, key):
            update_mask = stochastic_update(cell_states.shape, self.update_prob, key)
            new_states = cell_states + self.update_fn(perception) * update_mask
            if self.alive_index is not None:
                pre_alive = max_pool_alive(cell_states, self.alive_index, self.alive_threshold)
                post_alive = max_pool_alive(new_states, self.alive_index, self.alive_threshold)
                new_states = new_states * pre_alive * post_alive
            return new_states

        init_state = jax.nn.sigmoid(self.init_prepatterns[input_idx])
        init_state = jnp.concat([init_state, jnp.zeros((self.state_size - 4, *self.input_size))])

        x, dev_path = run_ca(
            init_state, perception_fn, update_fn, lambda *args, **kwargs: 0.0, self.num_dev_steps, key
        )
        return x[:4], dev_path

    def __str__(self) -> str:
        #NOTE: very illegible but essentially, pic the class name from the output of `type`
        return str(type(self)).split(' ')[-1].split('.')[-1][:-2]

