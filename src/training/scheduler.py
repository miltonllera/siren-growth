import jax
import optax


def cyclic_linear_schedule(
    min_lr: float,
    max_lr: float,
    half_cycle_length: int,
    scale: float = 1.0
) -> optax.Schedule:
    """
    Linear cyclic learning rate scheduler.

    The learning rate is linearly increased from `min_lr` to `max_lr` over the first half of the
    cycle, and then linearly decreased back to `min_lr` over the second half. This pattern is
    repeated for the duration of training. Corresponds to the "triangular" mode for cyclic learning
    rates used in PyTorch. Using a scaling less than 1.0 will reduce the learning rate after each
    cycle and is equivalent to the "triangular2" mode if scale=0.5.

    See: https://arxiv.org/pdf/1506.01186

    Args:
        min_lr (float): Minimum learning rate.
        max_lr (float): Maximum learning rate.
        half_cycle_length (int): Number of iterations in half a cycle.
        scale (float): Scaling factor for the learning rate after each cycle.

    """
    cycle_length = 2 * half_cycle_length

    linear_warmup = optax.linear_schedule(min_lr, max_lr, half_cycle_length)
    linear_decay = optax.linear_schedule(max_lr, min_lr, half_cycle_length)
    decay_from_cycle_mid_point = lambda iwc: linear_decay(iwc - half_cycle_length)

    def lr_schedule(global_step):
        num_cycles = global_step // cycle_length
        iter_within_cycle = global_step % cycle_length
        offset_from_min_lr = jax.lax.cond(
            iter_within_cycle < half_cycle_length,
            linear_warmup,
            decay_from_cycle_mid_point,
            iter_within_cycle
        )
        return min_lr + (offset_from_min_lr - min_lr) * (scale ** num_cycles)

    return lr_schedule


def warmup_then_cyclic_schedule(
    min_lr: float,
    max_lr: float,
    warmup_iters: int,
    half_cycle_length: int,
    scale: float = 1.0
) -> optax.Schedule:
    """
    Learning rate schedule with a linear warmup followed by a cyclic linear schedule.
    The learning rate is linearly increased from 0 to `max_lr` over `warmup_iters` iterations,
    and then follows a cyclic linear schedule between `min_lr` and `max_lr` as described in
    `cyclic_linear_schedule`.
    Args:
        min_lr (float): Minimum learning rate.
        max_lr (float): Maximum learning rate.
        warmup_iters (int): Number of iterations for the linear warmup.
        half_cycle_length (int): Number of iterations in half a cycle.
        scale (float): Scaling factor for the learning rate after each cycle.
    """
    cyclic_scheduler = cyclic_linear_schedule(min_lr, max_lr, half_cycle_length, scale)
    linear_warmup = optax.linear_schedule(0.0, max_lr, warmup_iters)
    def lr_schedule(global_step):
        return jax.lax.cond(
            global_step < warmup_iters,
            linear_warmup,
            cyclic_scheduler,
            global_step
        )
    return lr_schedule
