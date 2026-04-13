import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
from typing import Callable
from jaxtyping import Array, Float, Int


State = Float[Array, "S H W"]
Carry = tuple[State, jax.Array]


def sample_num_steps(n_steps: int | tuple[int, int] , key: jax.Array):
    if isinstance(n_steps, int):
        return n_steps, n_steps
    return jr.randint(key, (1,), *n_steps).squeeze(), n_steps[1]


def run_ca(
    init_state: State,
    perception_fn: Callable,
    update_fn: Callable,
    conditioning_fn: Callable,
    n_steps: int | tuple[int, int],
    key: jax.Array,
):
    carry_key , sample_key = jr.split(key)
    num_dev_steps, max_steps = sample_num_steps(n_steps, sample_key)

    def f(carry: Carry, step: Int) -> tuple[Carry, State]:
        cell_states, key = carry
        p_key, c_key, u_key, key = jr.split(key, 4)

        conditioning_vector = conditioning_fn(cell_states, key=c_key)
        perception = perception_fn(cell_states, conditioning_vector, key=p_key)
        updated_states = update_fn(cell_states, perception, key=u_key)

        cell_states = lax.select(step >= num_dev_steps, cell_states, updated_states)

        return (cell_states, key), cell_states

    _, dev_path = lax.scan(f, (init_state, carry_key), jnp.arange(max_steps))
    dev_path = jnp.concat([init_state[None], dev_path], axis=0)

    return dev_path[num_dev_steps], dev_path

