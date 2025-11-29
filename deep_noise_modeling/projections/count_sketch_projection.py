import jax
import jax.numpy as jnp
from flax import linen as nn


class CountSketch_Projection(nn.Module):
    """
    CountSketch projection:
        y_j = sum_{i : h(i) = j} s(i) * x_i

    Input:
        x: [..., input_dim]
    Output:
        y: [..., output_dim]

    Parameters stored (fixed after init):
        hash_idx: int32 array of shape (input_dim,)
        signs:    float32 array of shape (input_dim,) ∈ {+1, -1}

    Efficient implementation: no dense matrix and uses scatter_add.
    """
    input_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Type & shape check
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Last dimension {x.shape[-1]} != expected input_dim {self.input_dim}"
            )

        # --- 1. Initialize the hash function h(i) ---
        def hash_init(key, shape):
            # Random integers in [0, output_dim)
            return jax.random.randint(
                key, shape=shape, minval=0, maxval=self.output_dim, dtype=jnp.int32
            )

        hash_idx = self.param("hash_idx", hash_init, (self.input_dim,))

        # --- 2. Initialize random signs s(i) ∈ {+1, -1} ---
        def signs_init(key, shape):
            return 2.0 * jax.random.bernoulli(key, p=0.5, shape=shape) - 1.0

        signs = self.param("signs", signs_init, (self.input_dim,))

        # --- 3. Multiply input by signs (s(i) * x[i]) ---
        x_signed = x * signs

        # --- 4. Scatter-add into output buckets: y[h(i)] += x_signed[i] ---

        # Prepare zeros of shape (..., output_dim)
        y = jnp.zeros(x.shape[:-1] + (self.output_dim,), dtype=x.dtype)

        # To use scatter_add, expand hash_idx so it matches batch dims
        # If x has shape (B, input_dim), hash_idx is expanded to (B, input_dim)
        # via broadcasting along batch axes.
        # scatter indices must be int and same rank as x_signed.
        scatter_idx = jnp.broadcast_to(hash_idx, x_signed.shape)

        # scatter_add along the *last* axis (output buckets)
        y = y.at[..., scatter_idx].add(x_signed)

        return y
