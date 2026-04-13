from functools import partial
import jax
import jax.numpy as jnp
import equinox.nn as nn
from jaxtyping import Array


def conv2d_wrapped(inputs, kernel):
    inputs = jnp.pad(inputs, pad_width=1, mode='wrap')
    return jax.scipy.signal.convolve2d(inputs, kernel, mode='valid')


def conv2d_same_symm_break(inputs, kernel):
    inputs = jnp.pad(inputs, pad_width=1, mode='constant')
    #NOTE: we apply this per channel, so the row dimension is the first.
    inputs = inputs.at[0].set(-1)
    inputs = inputs.at[-1].set(-2)
    inputs = inputs.at[:, 0].set(-3)
    inputs = inputs.at[:, -1].set(-4)
    return jax.scipy.signal.convolve2d(inputs, kernel, mode='valid')


conv2d_same = partial(jax.scipy.signal.convolve2d, mode='same')

#------------------------------------------ Kernels ----------------------------------------------

sobel_x = jnp.array([
    [-1.0, 0.0, 1.0],
    [-2.0, 0.0, 2.0],
    [-1.0, 0.0, 1.0]
]) / 8.0

sobel_y = sobel_x.T

laplace = jnp.array([
    [1.0,   2.0, 1.0],
    [2.0, -12.0, 2.0],
    [1.0,   2.0, 1.0]
]) / 24.0


#------------------------------------ Perception functions ---------------------------------------

def sobel_perception(inputs: Array, use_laplace=False, key=None):
    x_conv = jax.vmap(conv2d_same, in_axes=(0, None))(inputs, sobel_x)
    y_conv = jax.vmap(conv2d_same, in_axes=(0, None))(inputs, sobel_y)

    if use_laplace:
        state_lap = jax.vmap(conv2d_same, in_axes=(0, None))(inputs, laplace)
        features = [inputs, x_conv, y_conv, state_lap]

    else:
        features = [inputs, x_conv, y_conv]

    return jnp.concat(features, axis=0)


def steerable_perception(inputs: Array, use_laplace=False, key=None):
    state, angle = inputs[:-1], inputs[-1:]

    x_conv = jax.vmap(conv2d_same, in_axes=(0, None))(state, sobel_x)
    y_conv = jax.vmap(conv2d_same, in_axes=(0, None))(state, sobel_y)

    c, s = jnp.cos(angle), jnp.sin(angle)

    rot_grad = jnp.concat([x_conv * c + y_conv * s, y_conv * c - x_conv * s], axis=0)

    if use_laplace:
        state_lap = jax.vmap(conv2d_same, in_axes=(0, None))(state, laplace)
        features = [state, rot_grad, state_lap]
    else:
        features = [state, rot_grad]

    return jnp.concat(features, axis=0)


def laplace_perception(inputs: Array, key=None, is_wrapped=True):
    conv2d = conv2d_wrapped if is_wrapped else conv2d_same_symm_break
    x = jax.vmap(conv2d, in_axes=(0, None))(inputs, laplace)
    return jnp.concat([inputs, x], axis=0)


#------------------------------------ Perception initialisation ----------------------------------

def init_perception(
    perception_type,
    state_size,
    key=None
):
    if 'sobel' in perception_type:
        return partial(sobel_perception, use_laplace='with-laplace' in perception_type)
    elif perception_type == 'laplace-same':
        return partial(laplace_perception, is_wrapped=False)
    elif perception_type == 'laplace-wrapped':
        return laplace_perception
    elif perception_type == 'learned':
        assert key is not None
        return nn.Conv2d(
            in_channels=state_size,
            out_channels=state_size,
            kernel_size=3,
            padding=1,
            padding_mode='ZEROS',
            groups=state_size,
            key=key
        )
    else:
        raise RuntimeError(f"Unrecognized perception type {perception_type}")
