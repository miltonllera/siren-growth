import warnings
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Sequence, overload
from jaxtyping import Array, Float

from equinox._custom_types import sentinel
from equinox.nn._stateful import State
from equinox.nn._misc import named_scope
from matplotlib.pyplot import axis


class InstanceNorm(eqx.Module):
    r"""
    Computes a mean and standard deviation over the whole input array, and uses these
    to normalise the whole array. Optionally applies an elementwise affine
    transformation afterwards.

    Given an input array $x$, this layer computes

    $$\frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \varepsilon}} * \gamma + \beta$$

    where $\gamma$, $\beta$ have the same shape as $x$ if `elementwise_affine=True`,
    and $\gamma = 1$, $\beta = 0$ if `elementwise_affine=False`.

    ??? cite
        [Layer Normalization](https://arxiv.org/abs/1607.06450)

        ```bibtex
        @article{ba2016layer,
            author={Jimmy Lei Ba, Jamie Ryan Kriso, Geoffrey E. Hinton},
            title={Layer Normalization},
            year={2016},
            journal={arXiv:1607.06450},
        }
        ```

    !!! faq "FAQ"

        If you need to normalise over only some input dimensions, then this can be
        achieved by vmap'ing. For example the following will compute statistics over
        every dimension *except* the first:
        ```python
        layer = LayerNorm(...)
        array = jax.vmap(layer)(array)
        ```

    """

    shape: tuple[int, ...] = eqx.field(static=True)
    axis: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    use_weight: bool = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    weight: Float[Array, "*shape"] | None
    bias: Float[Array, "*shape"] | None

    def __init__(
        self,
        shape: int | Sequence[int],
        axis: int = -1,
        eps: float = 1e-5,
        use_weight: bool = True,
        use_bias: bool = True,
        dtype=None,
        *,
        elementwise_affine: bool | None = None,
    ):
        """**Arguments:**

        - `shape`: Shape of the input.
        - `eps`: Value added to denominator for numerical stability.
        - `use_weight`: Whether the module has learnable affine weights.
        - `use_bias`: Whether the module has learnable affine biases.
        - `dtype`: The dtype to use for the weight and the bias in this layer if
            `use_weight` or `use_bias` is set to `True`.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `elementwise_affine`: Deprecated alternative to `use_weight` and `use_bias`.
        """
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(shape)
        self.shape = shape
        self.axis = axis
        self.eps = eps
        if elementwise_affine is not None:
            use_weight = elementwise_affine
            use_bias = elementwise_affine
            warnings.warn(
                "LayerNorm(elementwise_affine=...) is deprecated "
                "in favour of LayerNorm(use_weight=...) and LayerNorm(use_bias=...)"
            )
        self.use_weight = use_weight
        self.use_bias = use_bias
        self.weight = jnp.ones(shape, dtype=dtype) if use_weight else None
        self.bias = jnp.zeros(shape, dtype=dtype) if use_bias else None

    @overload
    def __call__(self, x: Array, *, key: jax.Array | None = None) -> Array: ...

    @overload
    def __call__(
        self, x: Array, state: State, *, key: jax.Array | None = None
    ) -> tuple[Array, State]: ...

    @named_scope("eqx.nn.LayerNorm")
    def __call__(
        self,
        x: Float[Array, "*shape"],
        state: State = sentinel,
        *,
        key: jax.Array | None = None,
    ) -> Array | tuple[jax.Array, State]:
        """**Arguments:**

        - `x`: A JAX array, with the same shape as the `shape` passed to `__init__`.
        - `state`: Ignored; provided for interchangeability with the
            [`equinox.nn.BatchNorm`][] API.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        The output is a JAX array of the same shape as `x`.

        If `state` is passed, then a 2-tuple of `(output, state)` is returned. The state
        is passed through unchanged. If `state` is not passed, then just the output is
        returned.
        """
        if x.shape != self.shape:
            raise ValueError(
                "`LayerNorm(shape)(x)` must satisfy the invariant `shape == x.shape`"
                f"Received `shape={self.shape} and `x.shape={x.shape}`. You might need "
                "to replace `layer_norm(x)` with `jax.vmap(layer_norm)(x)`.\n"
                "\n"
                "If this is a new error for you, it might be because this became "
                "stricter in Equinox v0.11.0. Previously all that was required is that "
                "`x.shape` ended with `shape`. However, this turned out to be a "
                "frequent source of bugs, so we made the check stricter!"
            )
        orig_dtype = x.dtype
        with jax.numpy_dtype_promotion("standard"):
            dtype = jnp.result_type(x.dtype, jnp.float32)

        x = x.astype(dtype)
        mean = jnp.mean(x, axis=self.axis, keepdims=True)
        variance = jnp.var(x, axis=self.axis, keepdims=True)
        variance = jnp.maximum(0.0, variance)
        inv = jax.lax.rsqrt(variance + self.eps)
        out = (x - mean) * inv
        if self.use_weight:
            out = self.weight.astype(dtype) * out  # pyright: ignore
        if self.use_bias:
            out = out + self.bias.astype(dtype)  # pyright: ignore
        if state is sentinel:
            return out.astype(orig_dtype)
        else:
            return out.astype(orig_dtype), state
