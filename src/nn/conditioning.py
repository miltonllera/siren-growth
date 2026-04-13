import functools as ft
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn
from jaxtyping import Array, Bool, Float
from equinox.nn._attention import dot_product_attention_weights


def dot_product_attention(
    query: Float[Array, "q_seq qk_size"],
    key_: Float[Array, "kv_seq qk_size"],
    value: Float[Array, "kv_seq v_size"],
    mask: Bool[Array, "q_seq kv_seq"] | None = None,
    dropout: eqx.nn.Dropout | None = None,
    *,
    key: jax.Array | None = None,
    inference: bool | None = None,
) -> tuple[Float[Array, "q_seq v_size"], Float[Array, "q_seq kv_seq"]]:
    weights = dot_product_attention_weights(query, key_, mask)
    if dropout is not None:
        weights = dropout(weights, key=key, inference=inference)
    attn = jnp.einsum("sS,Sd->sd", weights, value)
    return attn, weights


class MultiheadAttention(eqx.nn.MultiheadAttention):
    """
    A partial re-write of the MultiheadAttention module from Equinox.

    The original implementation does not return the attention weights as part of the module
    outputs. However, we wish to use these weights to understand what the calling module is
    attending to. The rewrite is thus very minimal, just adding an extra output to
    'dot_product_attention' and handling these appropriately in the main function call.
    """
    @jax.named_scope("eqx.nn.MultiheadAttention")
    def __call__(  # type: ignore
        self,
        query: Float[Array, "q_seq q_size"],
        key_: Float[Array, "kv_seq k_size"],
        value: Float[Array, "kv_seq v_size"],
        mask: Bool[Array, "q_seq kv_seq"] | Bool[Array, "num_heads q_seq kv_seq"] | None = None,
        *,
        key: jax.Array | None = None,
        inference: bool | None = None,
    ) -> tuple[Float[Array, "q_seq v_size"], Float[Array, "q_seq kv_seq"]]:
        query_seq_length, _ = query.shape
        kv_seq_length, _ = key_.shape
        kv_seq_length2, _ = value.shape

        if kv_seq_length != kv_seq_length2:
            # query length can be different
            raise ValueError("key and value must both be sequences of equal length.")

        query_heads = self._project(self.query_proj, query)
        key_heads = self._project(self.key_proj, key_)
        value_heads = self._project(self.value_proj, value)

        attn_fn = partial(
            dot_product_attention, dropout=self.dropout, inference=inference
        )

        keys = None if key is None else jax.random.split(key, query_heads.shape[1])
        if mask is not None and mask.ndim == 3:
            # Batch `mask` and `keys` down their 0-th dimension.
            attn, weights = jax.vmap(attn_fn, in_axes=1, out_axes=1)(
                query_heads, key_heads, value_heads, mask=mask, key=keys
            )
        else:
            # Batch `keys` down its 0-th dimension.
            attn, weights = jax.vmap(ft.partial(attn_fn, mask=mask), in_axes=1, out_axes=1)(
                query_heads, key_heads, value_heads, key=keys
            )
        attn = attn.reshape(query_seq_length, -1)
        weights = weights.reshape(query_seq_length, -1)  # check that this is correct

        return jax.vmap(self.output_proj)(attn), weights


class CrossAttentionConditioning(eqx.Module):
    mh_attn: MultiheadAttention
    alive_channel: int | None
    alive_threshold: float

    def __init__(
        self,
        state_size: int,
        prompt_emb_size: int,
        num_heads: int=1,
        dropout: float=0.0,
        alive_channel: int | None = 3,
        alive_threshold: float = 0.1,
        *,
        key
    ):
        self.alive_channel = alive_channel
        self.alive_threshold = alive_threshold
        self.mh_attn = MultiheadAttention(
            num_heads=num_heads,
            query_size=state_size,
            key_size=prompt_emb_size,
            value_size=prompt_emb_size,
            output_size=state_size,
            qk_size=prompt_emb_size,
            dropout_p=0.0,  # type: ignore
            key=key, # type: ignore
        )

    def __call__(self, inputs, conditioning, key):
        if self.alive_channel is not None:
            alive = inputs[self.alive_channel:self.alive_channel+1] > self.alive_threshold
        else:
            alive = 1.0
        outputs, _ = self.mh_attn(inputs[None], conditioning, conditioning, key=key)
        return outputs[0] * alive  # since we only have one element in the query sequence (the cell)


class LinearConditioning(eqx.Module):
    """
    A conditioning function that applies a linear transformation to the input vector.
    This is useful for conditioning the cellular automata on a vector input, such as
    a prompt embedding.
    """
    linear: nn.Linear
    alive_channel: int | None
    alive_threshold: float

    def __init__(
        self,
        state_size: int,
        prompt_emb_size: int,
        alive_channel: int | None,
        alive_threshold: float,
        *,
        key
    ):
        self.linear = nn.Linear(prompt_emb_size + state_size, state_size, use_bias=False, key=key)
        self.alive_channel = alive_channel
        self.alive_threshold = alive_threshold

    def __call__(self, inputs, conditioning, key):
        if self.alive_channel is not None:
            alive = inputs[self.alive_channel:self.alive_channel+1] > self.alive_threshold
        else:
            alive = 1.0
        return self.linear(jnp.concat([conditioning, inputs])) * alive


class FactorLinearConditioning(eqx.Module):
    values_per_factor: list[int]
    split_indices: list[int]
    factor_transforms: list[nn.Linear]
    proj: nn.Linear
    alive_index: int | None
    alive_threshold: float

    def __init__(
        self,
        values_per_factor: list[int],
        output_size: int,
        alive_channel: int | None = 3,
        alive_threshold: float = 0.1,
        *,
        key
    ):
        one_hot_transforms = []
        for fv in values_per_factor:
            key, linear_key = jr.split(key)
            one_hot_transforms.append(nn.Linear(fv, output_size, key=linear_key))

        self.values_per_factor = values_per_factor
        self.split_indices = [int(v) for v in np.cumsum(np.asarray(values_per_factor))]
        self.factor_transforms = one_hot_transforms
        self.proj = nn.Linear(len(values_per_factor) * output_size, output_size, key=key)
        self.alive_index = alive_channel
        self.alive_threshold = alive_threshold

    def __call__(self, inputs, conditioning, key):
        conditioning = jnp.split(conditioning, self.split_indices)
        embeddings = []
        for i, f in zip(conditioning, self.factor_transforms):
            embeddings.append(f(i))

        # embeddings = jnp.stack(embeddings).sum(0)
        embeddings = jnp.concatenate(embeddings)

        if self.alive_index is not None:
            alive = inputs[self.alive_index:self.alive_index+1] > self.alive_threshold
        else:
            alive = 1.0

        return self.proj(embeddings) * alive

