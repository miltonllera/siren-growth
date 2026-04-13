from typing import Callable, Literal
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn
from jaxtyping import Float, Array
# from .norm import InstanceNorm


def stochastic_update(state_shape: tuple[int, ...], update_prob: float, key: jax.Array):
    _, H, W = state_shape
    return jnp.floor(jr.uniform(key, (1, H, W)) + update_prob)


def max_pool_alive(state: Float[Array, "C H W"], alive_index, alive_threshold):
    alive_value = jnp.pad(state[alive_index], pad_width=(1, 1), mode='wrap')[None]
    max_alive = nn.MaxPool2d(kernel_size=3, padding=0)(alive_value)
    return (max_alive  > alive_threshold)


def init_update_fn(
    perception_size: int,
    update_width: int,
    update_depth: int,
    state_size: int,
    pre_process_fn: eqx.Module | None,
    key: jax.Array,
):
    conv_key, update_key = jr.split(key)
    layers: list[eqx.Module] = [pre_process_fn] if pre_process_fn is not None else []

    layer_input_size = perception_size
    for _ in range(update_depth):
        update_key, conv_key = jr.split(update_key)
        layers.extend([
            nn.Conv2d(layer_input_size, update_width, kernel_size=1, key=conv_key),
            nn.Lambda(jax.nn.relu),
        ])
        layer_input_size = update_width

    layers.append(
        nn.Conv2d(layer_input_size, state_size, kernel_size=1, use_bias=False, key=update_key)
    )

    layers[-1] = eqx.tree_at(
        where=lambda l: l.weight,
        pytree=layers[-1],
        replace_fn=lambda w: jnp.zeros_like(w)
    )

    return nn.Sequential(layers)  # type: ignore


class GRUUpdate(eqx.Module):
    gru: nn.GRUCell
    n_steps: int

    def __init__(self, n_channels, state_size, n_steps=3, *, key):
        super().__init__()
        self.gru = nn.GRUCell(n_channels, state_size, use_bias=False, key=key)
        self.n_steps = n_steps

    def __call__(self, perception: Float[Array, "P H W"], state: Float[Array, "S H W"], key=None):
        perception_size, spatial_shape = perception.shape[0], perception.shape[1:]
        state_size = state.shape[0]

        perception = perception.reshape(perception_size, -1).T
        state = state.reshape(state_size, -1).T

        def f(state, step):
            carry = jax.vmap(self.gru)(perception, state)
            return carry, carry
        updated_state, _ = jax.lax.scan(f, state, length=self.n_steps)

        return updated_state.T.reshape(state_size, *spatial_shape)
