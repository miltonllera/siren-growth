import argparse
import json
import logging
from pathlib import Path

import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from grain import DataLoader
from grain.samplers import IndexSampler
from grain.transforms import Batch
from tqdm import tqdm, trange
from typing import Callable
from jaxtyping import Array, Float, PyTree

from src.dataset.emojis import EmojiDataset
from src.model.siren import SIREN
from src.nn.backbone import init_backbone
from src.training.checkpoint import CheckpointManager
from src.training.scheduler import cyclic_linear_schedule
from src.visualisation.utils import plot_examples
from src.utils.core import (
    cycle,
    filter_put,
    format_float,
    get_sharding_specs,
    init_wandb,
    seed_everything,
)


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def main(
    dataset_name: str,
    input_size: int,
    padding: int,
    latent_size: int,
    siren_width: int,
    siren_depth: int,
    batch_size: int,
    training_iters: int,
    val_freq: int,
    val_ratio: float,
    val_batch_size: int | None,
    learning_rate: float,
    use_lr_schedule: bool = False,
    schedule_half_cycle: int = 2000,
    cycle_scale: float = 1.0,
    save_intermediate: bool = False,
    save_folder: str | Path = 'data/logs/temp',
    seed: int | None = None,
    *,
    run,
):
    config = {k: v for k, v in locals().items() if k != 'run'}
    config['model_type'] = 'siren'

    # setup
    rng, jax_key = seed_everything(seed)

    # Loading dataset
    _logger.info("Loading dataset...")

    input_size = input_size, input_size  # type: ignore
    padding = padding, padding  # type: ignore

    if dataset_name == 'emojis':
        train_ds = val_ds = EmojiDataset("data/datasets/emojis", 'all', input_size, padding)
    else:
        raise NotImplementedError()

    train_loader = DataLoader(
        data_source=train_ds,
        operations=[Batch(batch_size, drop_remainder=True)],
        sampler=IndexSampler(len(train_ds), shuffle=True, seed=rng.choice(2 ** 32 - 1))
    )

    val_loader = DataLoader(
        data_source=val_ds,
        operations=[Batch(1)],
        sampler=IndexSampler(len(val_ds), shuffle=False, num_epochs=1)
    )

    # Initializing model
    _logger.info("Done. Initialising model...")

    input_shape = train_ds.input_shape
    encoder_backbone = init_backbone(input_shape, latent_size, dataset_name, key=jax_key)
    siren = SIREN(
        input_shape=input_shape,
        encoder_backbone=encoder_backbone,
        latent_size=latent_size,
        siren_width=siren_width,
        siren_depth=siren_depth,
        key=jax_key,
    )

    param_count = sum(x.size for x in jax.tree.leaves(siren) if isinstance(x, jax.Array))
    encoder_param_count = sum(
        x.size for x in jax.tree.leaves(siren.encoder) if isinstance(x, jax.Array)
    )
    _logger.info(
        f"Initialised model {str(siren)}.Number of parameters:"
        f"\n\tTotal: {param_count}."
        f"\n\t{str(siren)}: {str(param_count - encoder_param_count)}."
    )

    # Setting up training
    _logger.info("Setting up training...")
    if use_lr_schedule:
        schedule = cyclic_linear_schedule(
            min_lr=1e-7,
            max_lr=learning_rate,
            half_cycle_length=schedule_half_cycle,
            scale=cycle_scale
        )
    else:
        schedule = optax.constant_schedule(learning_rate)

    # optim = optax.chain(
    #     # clipping
    #     optax.clip(1.0),
    #     # optax.clip_by_global_norm(1.0),
    #     optax.adam(schedule, b1=0.9, b2=0.999),
    # )
    optim = optax.chain(
        # clipping
        optax.clip_by_block_rms(1.0),
        optax.sgd(schedule, nesterov=True, momentum=0.95)
    )
    opt_state = optim.init(eqx.filter(siren, eqx.is_array))

    def compute_loss(
        model: Callable,
        batch: tuple[Float[Array, "BSE"], Float[Array, "B4HW"]],
        key: jax.Array
    ) -> Float:
        inputs, _ = batch
        preds, _, _ = jax.vmap(model)(inputs, jr.split(key, len(inputs)))
        return jnp.sum(optax.l2_loss(preds, inputs)) / len(inputs)

    @eqx.filter_jit(donate='all')
    def train_step(
        model: PyTree,
        opt_state: PyTree,
        batch: tuple[Float[Array, "BSE"], Float[Array, "B4HW"]],
        key: jax.Array,
    ) -> tuple[PyTree, PyTree, Float]:
        loss_value, grads = eqx.filter_value_and_grad(compute_loss)(model, batch, key)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    @eqx.filter_jit(donate='all-except-first')
    def val_step(
        model: PyTree,
        batch: tuple[Float[Array, "BSE"], Float[Array, "B4HW"]],
        key: jax.Array
    ) -> Float:
        inputs, _ = batch
        max_steps_eval = lambda i, b: model(i, b)
        preds, _, _ = jax.vmap(max_steps_eval)(inputs, jr.split(key, len(inputs)))
        loss = jnp.sum(optax.l2_loss(preds, inputs))
        return loss

    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    (save_folder / 'config.json').write_text(json.dumps(config, indent=2, default=str))

    def plot_model_recons(model, suffix):
        inputs = np.stack([val_ds[i][0] for i in range(min(len(val_ds), 20))])
        recons, _, _ = jax.vmap(model)(inputs, jr.split(jax_key, len(inputs)))
        fig = plot_examples(recons, w=4, format='NCHW')
        fig.savefig( save_folder / f"examples_{suffix}.png")
        plt.close(fig)

    # Setting up checkpoint manager
    best_ckpt_mng = CheckpointManager(save_folder, mode='min', best_fn='val_mse', frozen=False)
    best_ckpt_mng.init_(reset=True)  # WARNING: This will erase the folder
    # periodic_ckpt_mng = CheckpointManager(save_folder, save_freq=1)

    # Sharding
    batch_sharding, model_sharding = get_sharding_specs()
    if batch_sharding is not None:
        shard_batch = lambda b: filter_put(b, batch_sharding)
    else:
        shard_batch = lambda b: b

    if model_sharding is not None:
        siren = filter_put(siren, model_sharding)

    # Training
    _logger.info("Done. Training is starting...")

    for i, batch in zip(pbar := trange(training_iters), cycle(train_loader)):
        jax_key, step_key = jr.split(jax_key)
        siren, opt_state, train_loss = train_step(siren, opt_state, shard_batch(batch), step_key)
        train_loss, lr_at_step = train_loss.item(), schedule(i)
        pbar.set_postfix_str(
            f"iter: {i}; loss: {format_float(train_loss)}, lr: {format_float(lr_at_step)}"
        )
        run.log({"train_mse": train_loss, "lr": lr_at_step})

        if (i + 1) % val_freq == 0:
            val_loss, total_examples, val_key = 0.0, 0, step_key
            for batch in tqdm(val_loader, leave=False):
                val_key, step_key = jr.split(val_key)
                val_loss += val_step(siren, batch, step_key).item()
                total_examples += len(batch[0])
            val_loss /= total_examples
            best_ckpt_mng.save(i, siren, {'val_mse': val_loss})
            # periodic_ckpt_mng.save(i, siren)
            if save_intermediate:
                plot_model_recons(siren, f"iter-{i}")
            run.log({"val_mse": val_loss})

    # Evaluating model
    _logger.info("Training completed. Evaluating best model...")
    best_model = best_ckpt_mng.best_ckpt(siren)
    plot_model_recons(best_model, "best")
    _logger.info("Done. Script is terminating.")


if __name__ == "__main__":
    def maybe_string(s):
        s = str(s)
        return None if s == 'none' else s

    parser = argparse.ArgumentParser(description="Train NCA model.")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default='flags',
        help="Path to the dataset file."
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=60,
        help="Spatial size of input images. Assumes square inputs (e.g. --input_size 64).",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=2,
        help="Spatial size of padding to be applied on each side for height and width"
    )
    parser.add_argument(
        "--latent_size",
        type=int,
        default=64,
        help="Size of the latent representation used to condition the NCA.",
    )
    parser.add_argument(
        "--siren_width",
        type=int,
        default=64,
        help="Width of the SIREN model used to initialise the developmental canvas."
    )
    parser.add_argument(
        "--siren_depth",
        type=int,
        default=2,
        help="Depth of the SIREN model used to initialise the developmental canvas.",
    )
    parser.add_argument(
        "--batch_size", "--bs",
        type=int,
        default=4,
        help="Batch size for training."
    )
    parser.add_argument(
        "--training_iters", "--ti",
        type=int,
        default=1000,
        help="Number of training iterations."
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of validation samples used"
    )
    parser.add_argument(
        "--val_freq",
        type=int,
        default=100,
        help="Frequency (in iterations) of validation loops."
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=None,
        help="Batch size for validation. If None, it will be set to training batch size."
    )
    parser.add_argument(
        "--learning_rate", "--lr",
        type=float,
        default=1e-5,
        help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--use_lr_schedule",
        action='store_true',
        help="Whether to use a learning rate schedule."
    )
    parser.add_argument(
        "--schedule_half_cycle",
        type=int,
        default=2000,
        help="Duration of half a cycle of the linear schedule."
    )
    parser.add_argument(
        "--cycle_scale",
        type=float,
        help="Scaling value for the learning rate after each cycle. A value of 1.0 produces " \
            " a triangular schedule. A value of 0.5, the trinagular2 mode"
    )
    parser.add_argument(
        "--save_intermediate",
        action='store_true',
        help="Whether to plot reconstruction examples throughout training."
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default='data/logs/temp',
        help="Folder to save the training logs and model checkpoints."
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        default=False,
    )

    args = vars(parser.parse_args())
    if args.pop('debug'):
        args['save_folder'] = "data/logs/temp"
        wandb_mode = 'disabled'
        _logger.info("Running in debug mode. No logging to wandb.")
    else:
        wandb_mode = 'online'

    with init_wandb(mode=wandb_mode, **args) as run:
        main(**args, run=run)

