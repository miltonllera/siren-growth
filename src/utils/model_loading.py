"""Utilities for reconstructing models from saved run folders and loading checkpoints.

The functions here mirror the helpers in notebooks/figures.ipynb but support
only NCA and SIREN_NCA (not SIREN), and read all hyperparameters from the
saved config rather than hard-coding defaults.
"""

from pathlib import Path

import jax
import jax.random as jr
import equinox as eqx
import orbax.checkpoint as ocp

from src.model.nca import NCA
from src.model.pm_nca import SIREN_NCA
from src.nn.backbone import init_backbone


def init_model(cfg: dict, key: jax.Array) -> eqx.Module:
    """Reconstruct a model from a saved config dict.

    Supports model_type values: 'pmnca' (SIREN_NCA) and 'nca' (NCA).

    The input_shape is derived from cfg['input_size'] and cfg['padding'] using
    the same convention as the training scripts:
        H = W = input_size + 2 * padding   (padding applied once per side)
        C = 4  (all supported datasets return RGBA)
    """
    dataset_name = cfg['dataset_name']
    latent_size = cfg['latent_size']

    input_size = cfg['input_size']
    padding = cfg['padding']
    # In training scripts both values are stored as plain ints before tuple conversion
    if isinstance(input_size, (list, tuple)):
        input_size = input_size[0]
    if isinstance(padding, (list, tuple)):
        padding = padding[0]
    H = W = input_size + 2 * padding
    C = 4
    input_shape = (C, H, W)

    dev_steps = cfg['dev_steps']
    if isinstance(dev_steps, list):
        dev_steps = tuple(dev_steps)
    elif isinstance(dev_steps, int):
        dev_steps = (dev_steps,)

    enc_key, model_key = jr.split(key)
    backbone = init_backbone(input_shape, latent_size, variant=dataset_name, key=enc_key)

    model_type = cfg['model_type']

    if model_type == 'pmnca':
        return SIREN_NCA(
            input_shape=input_shape,
            encoder_backbone=backbone,
            latent_size=latent_size,
            siren_width=cfg['siren_width'],
            siren_depth=cfg['siren_depth'],
            nca_hidden_size=cfg['hidden_state'],
            perception_type=cfg.get('perception_type', 'laplace-wrapped'),
            update_width=cfg['update_width'],
            update_depth=cfg['update_depth'],
            update_prob=cfg.get('update_prob', 1.0),
            alive_index=3 if cfg.get('growing', False) else None,
            num_dev_steps=dev_steps,
            key=model_key,
        )
    elif model_type == 'nca':
        return NCA(
            input_shape=input_shape,
            hidden_size=cfg['hidden_state'],
            encoder_backbone=backbone,
            latent_size=latent_size,
            perception_type=cfg.get('perception_type', 'sobel-with-laplace'),
            update_width=cfg['update_width'],
            update_depth=cfg['update_depth'],
            update_prob=cfg.get('update_prob', 1.0),
            conditioning_mode=cfg.get('conditioning_mode', 'concat'),
            alive_index=3 if cfg.get('growing', False) else None,
            num_dev_steps=dev_steps,
            key=model_key,
        )
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Expected 'pmnca' or 'nca'.")


def load_best_checkpoint(run_dir: str | Path, abstract_model: eqx.Module) -> eqx.Module:
    """Load the best checkpoint from a run directory into abstract_model.

    Matches the Orbax usage in notebooks/figures.ipynb: looks for a checkpoint
    with step_prefix="best" as written by CheckpointManager(best_fn=...).
    """
    params, static = eqx.partition(abstract_model, eqx.is_array)
    with ocp.CheckpointManager(
        Path(run_dir).absolute() / "checkpoints",
        options=ocp.CheckpointManagerOptions(step_prefix="best"),
    ) as mngr:
        step = mngr.latest_step()
        if step is None:
            raise RuntimeError(f"No checkpoint found in {run_dir}/checkpoints/")
        params = mngr.restore(step, args=ocp.args.StandardRestore(params))
    return eqx.combine(params, static)
