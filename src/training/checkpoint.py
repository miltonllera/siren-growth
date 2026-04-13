import equinox as eqx
import orbax.checkpoint as ocp
from pathlib import Path
from typing import Callable
from jaxtyping import PyTree


class CheckpointManager:
    """
    A simple wrapper around Orbax's `CheckpointManager` which works with Equinox models.

    This avoids having to import and manually call the different components that Orbax uses to
    initalise a checkpoint manager. Note that `best_fn` can be a string, callable or None. The
    first option indicates that metrics will be a dictionary with string keys, while None will
    save checkpoints based on recency. The rest of the parameters pretty much map with to ones
    used by `CheckpointManagerOptions` one-to-one.
    """
    def __init__(
        self,
        run_dir: str | Path,
        save_freq: int = 1,
        mode: str = 'min',
        best_fn: str | Callable | None = None,
        max_checkpoints: int = 1,
        frozen: bool = False,
    ):
        if isinstance(best_fn, str):
            _best_fn = lambda metrics: metrics[best_fn]
        else:
            _best_fn = best_fn

        self.run_dir = Path(run_dir).absolute()
        self.manager = ocp.CheckpointManager(
            directory=self.run_dir / "checkpoints",
            options=ocp.CheckpointManagerOptions(
                step_prefix='best' if best_fn is not None else 'periodic',
                max_to_keep=max_checkpoints,
                save_interval_steps=save_freq,
                best_fn=_best_fn,
                best_mode=mode,
            ),
        )
        self.frozen = frozen

    def __del__(self):
        self.manager.wait_until_finished()
        self.manager.close()

    def init_(self, reset=False):
        if self.frozen:
            raise RuntimeError("Cannot initialise checkpoint. Manager is frozen.")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if not self.manager.directory.exists() or reset:
            ocp.test_utils.erase_and_create_empty(self.manager.directory)

    def save(self, step: int, model: PyTree, metrics: dict[str, float] | None = None):
        if self.frozen:
            raise RuntimeError("Checkpoint is frozen, cannot initialise.")

        params = eqx.filter(model, eqx.is_array)
        self.manager.save(
            step,
            args=ocp.args.StandardSave(params),  # type: ignore
            metrics=metrics,
        )

    def restore(self, step: int, abstract_pytree: PyTree):
        params, static = eqx.partition(abstract_pytree, eqx.is_array)
        params = self.manager.restore(
            step, args=ocp.args.StandardRestore(params) # type: ignore
        )
        return eqx.combine(params, static)

    def best_ckpt(self, abstract_pytree: PyTree):
        step = self.manager.best_step()
        if step is None:
            raise RuntimeError("No checkpoint to load.")
        return self.restore(step, abstract_pytree)

