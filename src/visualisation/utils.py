import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from jaxtyping import Array, Float


def strip(ax=None):
    if ax == None:
        ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def tile2d(a, w=None, format='NCWH'):
    a = np.asarray(a)

    if format == 'NCHW':
        a = a.transpose(0, 2, 3, 1)

    if w is None:
        w = int(np.ceil(np.sqrt(a.shape[0])))

    th, tw = a.shape[1:3]
    pad = (w - a.shape[0]) % w
    pad_value = 1.0 if np.issubdtype(a.dtype, np.floating) else 0
    a = np.pad(a, [(0, pad)]+[(0, 0)]*(a.ndim-1), 'constant', constant_values=pad_value)
    h = a.shape[0] // w
    a = a.reshape([h, w]+list(a.shape[1:]))
    a = np.rollaxis(a, 2, 1).reshape([th*h, tw*w]+list(a.shape[4:]))

    if format == 'NCHW':
        a = a.transpose(2, 0, 1)

    return a


#----------------------------------------------- Plots -------------------------------------------

def plot_examples(a, w=None, format='NCHW', ax=None):
    a = np.clip(a, 0, 1)
    if format=='NCHW':
        a = a.transpose(0, 2, 3, 1)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    a = tile2d(a, w, format='NHWC')
    ax.imshow(a)
    strip(ax)
    return fig


def plot_dev_path(dev_path: Float[Array, "C H W"]):
    fig = plt.figure()
    ax = plt.gca()
    strip(ax)

    dev_path = dev_path.transpose(0, 2, 3, 1).clip(0.0, 1.0)

    im = plt.imshow(dev_path[0], vmin=0, vmax=1)
    def animate(i):
        ax.set_title(f"Growth step: {i}")
        im.set_array(dev_path[i])
        return im,

    ani = FuncAnimation(fig, animate, interval=200, blit=True, repeat=True, frames=len(dev_path))
    plt.close(fig)
    return ani


def plot_batch_dev_paths(dev_paths, n_cols: int = 4):
    """Create a grid animation from a list of dev_path arrays (each [T, C, H, W])."""
    import math

    n = len(dev_paths)
    n_cols = min(n_cols, n)
    n_rows = math.ceil(n / n_cols)

    dev_paths = [np.asarray(dp).transpose(0, 2, 3, 1).clip(0.0, 1.0) for dp in dev_paths]
    n_frames = len(dev_paths[0])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = np.array(axes).reshape(n_rows, n_cols)

    ims = []
    for dp, ax in zip(dev_paths, axes.flat):
        strip(ax)
        im = ax.imshow(dp[0], vmin=0, vmax=1)
        ims.append((im, ax, dp))

    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)

    def animate(frame):
        for im, ax, dp in ims:
            ax.set_title(f"Step: {frame}", fontsize=8)
            im.set_array(dp[frame])
        return [im for im, _, _ in ims]

    ani = FuncAnimation(fig, animate, interval=200, blit=True, repeat=True, frames=n_frames)
    plt.close(fig)
    return ani


