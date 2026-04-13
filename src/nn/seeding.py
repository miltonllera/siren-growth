import numpy as np
import jax.random as jr


def init_central_seed(shape, key=None):
    _, H, W = shape
    init = np.zeros(shape)
    init[3:, H//2, W//2] = 1.0
    return init


def init_random(shape, key):
    return jr.uniform(key, shape, minval=-1, maxval=1)


def init_zeros(shape, key=None):
    return np.zeros(shape)

def _grid(H, W):
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def mask_circle(shape, radius=0.8):
    _, H, W = shape
    xx, yy = _grid(H, W)
    mask = (xx**2 + yy**2 <= radius**2).astype(np.float32)
    return np.broadcast_to(mask[None], shape)


def mask_oval(shape, radius_x=0.8, radius_y=0.5):
    H, W = shape
    xx, yy = _grid(H, W)
    mask = ((xx / radius_x)**2 + (yy / radius_y)**2 <= 1.0).astype(np.float32)
    return mask


def mask_triangle(shape, base=1.6, height=1.6):
    _, H, W = shape
    xx, yy = _grid(H, W)
    half_base = base / 2
    y_top = 1.0 - (2.0 - height) / 2
    y_bot = y_top - height
    t = (yy - y_bot) / height
    half_width = half_base * (1.0 - t)
    mask = ((yy >= y_bot) & (yy <= y_top) & (np.abs(xx) <= half_width)).astype(np.float32)
    return np.broadcast_to(mask[None], shape)


def get_mask_fn(mask_type):
    if mask_type == 'circle':
        return mask_circle
    elif mask_type == 'oval':
        return mask_oval
    elif mask_type == 'triangle':
        return mask_triangle
    elif mask_type is None:
        return lambda *_a, **_kw: None
    else:
        raise RuntimeError(f"Unknown mask type: {mask_type}")


def get_init_fn(init_type):
    if init_type == 'central':
        return init_central_seed
    elif init_type == 'random':
        return init_random
    elif init_type == 'empty':
        return init_zeros
    else:
        raise RuntimeError()
