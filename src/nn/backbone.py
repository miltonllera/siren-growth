import jax
import jax.random as jr
import equinox.nn as nn

def check_size(x):
    print(x.shape)
    return x

def init_backbone(input_shape, latent_size, variant, *, key):
    if variant == 'flags' and input_shape[1] == 16:
        keys = jr.split(key, 3)
        return nn.Sequential([
            nn.Conv2d(
                input_shape[0],
                out_channels=16,
                kernel_size=2,
                padding=0,
                stride=2,
                key=keys[0],
            ),
            nn.Lambda(jax.nn.relu),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=2,
                padding=0,
                stride=2,
                key=keys[1],
            ),
            nn.Lambda(jax.nn.relu),
            nn.Lambda(lambda x: x.reshape(-1)),
            nn.Linear(
                in_features=256,
                out_features=latent_size,
                key=keys[2]
            ),
            nn.Lambda(jax.nn.relu),
        ])

    elif variant == 'flags' and input_shape[1] == 32:
        keys = jr.split(key, 3)
        return nn.Sequential([
            nn.Conv2d(
                input_shape[0],
                out_channels=16,
                kernel_size=4,
                padding=0,
                stride=4,
                key=keys[0],
            ),
            nn.Lambda(jax.nn.relu),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=2,
                padding=0,
                stride=2,
                key=keys[1],
            ),
            nn.Lambda(jax.nn.relu),
            nn.Lambda(lambda x: x.reshape(-1)),
            nn.Linear(
                in_features=256,
                out_features=latent_size,
                key=keys[2]
            ),
            nn.Lambda(jax.nn.relu),
        ])

    elif variant == 'flags' and input_shape[1] == 64:
        keys = jr.split(key, 3)
        return nn.Sequential([
            nn.Conv2d(
                input_shape[0],
                out_channels=16,
                kernel_size=4,
                padding=0,
                stride=4,
                key=keys[0],
            ),
            nn.Lambda(jax.nn.elu),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=4,
                padding=0,
                stride=4,
                key=keys[1],
            ),
            nn.Lambda(jax.nn.elu),
            nn.Lambda(lambda x: x.reshape(-1)),
            nn.Linear(
                in_features=256,
                out_features=latent_size,
                key=keys[2]
            ),
            nn.Lambda(jax.nn.elu),
        ])

    elif variant == 'emojis' and input_shape[1] == 48:
        keys = jr.split(key, 3)
        return nn.Sequential([
            nn.Conv2d(
                input_shape[0],
                out_channels=16,
                kernel_size=4,
                padding=0,
                stride=4,
                key=keys[0],
            ),
            nn.Lambda(jax.nn.elu),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=2,
                padding=0,
                stride=2,
                key=keys[1],
            ),
            nn.Lambda(jax.nn.elu),
            nn.Lambda(lambda x: x.reshape(-1)),
            nn.Linear(
                in_features=576,
                out_features=latent_size,
                key=keys[2]
            ),
            nn.Lambda(jax.nn.elu),
        ])
    elif variant in ['emojis', 'animoji'] and input_shape[1] == 64:
        keys = jr.split(key, 3)
        return nn.Sequential([
            nn.Conv2d(
                input_shape[0],
                out_channels=16,
                kernel_size=2,
                padding=0,
                stride=2,
                key=keys[0],
            ),
            nn.Lambda(jax.nn.elu),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=2,
                padding=0,
                stride=2,
                key=keys[1],
            ),
            nn.Lambda(jax.nn.elu),
            nn.Conv2d(
                in_channels=16,
                out_channels=8,
                kernel_size=2,
                padding=0,
                stride=2,
                key=keys[1],
            ),
            nn.Lambda(jax.nn.elu),
            nn.Lambda(lambda x: x.reshape(-1)),
            nn.Linear(
                in_features=512,
                out_features=latent_size,
                key=keys[2]
            ),
            nn.Lambda(jax.nn.elu),
        ])

    elif variant == 'emojis' and input_shape[1] == 96:
        keys = jr.split(key, 3)
        return nn.Sequential([
            nn.Conv2d(
                input_shape[0],
                out_channels=16,
                kernel_size=2,
                padding=0,
                stride=2,
                key=keys[0],
            ),
            nn.Lambda(jax.nn.elu),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=2,
                padding=0,
                stride=2,
                key=keys[1],
            ),
            nn.Lambda(jax.nn.elu),
            nn.Conv2d(
                in_channels=16,
                out_channels=8,
                kernel_size=2,
                padding=0,
                stride=2,
                key=keys[1],
            ),
            nn.Lambda(jax.nn.elu),
            nn.Lambda(lambda x: x.reshape(-1)),
            nn.Linear(
                in_features=1152,
                out_features=latent_size,
                key=keys[2]
            ),
            nn.Lambda(jax.nn.elu),
        ])

    elif variant == 'celeb_a' and input_shape[1] == 64:
        keys = jr.split(key, 3)
        return nn.Sequential([
            nn.Conv2d(
                input_shape[0],
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                key=keys[0],
            ),
            nn.Lambda(jax.nn.elu),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                key=keys[1],
            ),
            nn.Lambda(jax.nn.elu),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                key=keys[1],
            ),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                key=keys[1],
            ),
            nn.Lambda(jax.nn.elu),
            nn.Lambda(lambda x: x.reshape(-1)),
            nn.Linear(
                in_features=512,
                out_features=latent_size,
                key=keys[2]
            ),
            nn.Lambda(jax.nn.elu),
        ])

    else:
        raise NotImplementedError()
