import math
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return 1 << (n - 1).bit_length()


def fwht(x: jnp.ndarray) -> jnp.ndarray:
    """
    Fast Walsh–Hadamard Transform (FWHT) along the last axis.
    Assumes the last dimension is a power of 2.
    Returns an orthonormal transform: H / sqrt(n).
    """
    n = x.shape[-1]
    # Require power of 2
    if n & (n - 1) != 0:
        raise ValueError(f"FWHT requires power-of-two length, got {n}")

    batch_shape = x.shape[:-1]
    h = 1
    # Iterative butterfly
    while h < n:
        # Reshape to (..., num_blocks, 2, h)
        # We want to group pairs of blocks of size h.
        # The last dimension n is viewed as (n // (2*h), 2, h)
        x = x.reshape(*batch_shape, -1, 2, h)
        
        x_even = x[..., 0, :]
        x_odd = x[..., 1, :]
        
        # Butterfly: [x_even + x_odd, x_even - x_odd]
        # Concatenate along the last axis (h) -> (..., num_blocks, 2*h)
        x = jnp.concatenate([x_even + x_odd, x_even - x_odd], axis=-1)
        
        h *= 2

    # Reshape back to (*batch_shape, n)
    x = x.reshape(*batch_shape, n)

    # Orthonormal scaling
    return x / jnp.sqrt(n)


class SRHT_Projection_Padded(nn.Module):
    """
    SRHT projection with automatic zero-padding to the next power of 2.

    Implements: y = (1/sqrt(k)) * S * H * D * x_padded

    - input_dim:  original input dimension (can be any positive int)
    - output_dim: projected dimension k

    Internally:
      padded_dim = next_power_of_two(input_dim)
      diag   ~ {+1, -1}^padded_dim  (random sign flips, D)
      indices ~ random subset of {0, ..., padded_dim-1} of size output_dim (S)
    """
    input_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: [..., input_dim]
        returns: [..., output_dim]
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Last dimension of x ({x.shape[-1]}) must equal input_dim ({self.input_dim})"
            )

        padded_dim = _next_power_of_two(self.input_dim)

        # --- Initialize SRHT "parameters" (random but typically kept fixed) ---

        # Random ±1 diagonal D (length = padded_dim)
        def diag_init(key, shape):
            # Bernoulli({0,1}) -> {−1,+1}
            return 2.0 * jax.random.bernoulli(key, p=0.5, shape=shape) - 1.0

        diag = self.param("diag", diag_init, (padded_dim,))

        # Subsampling indices S: choose output_dim rows from {0,...,padded_dim-1}
        def indices_init(key, shape):
            return jax.random.choice(
                key,
                padded_dim,
                shape=shape,
                replace=False,  # typical SRHT uses sampling without replacement
            )

        indices = self.param("indices", indices_init, (self.output_dim,))

        # --- 1. Zero-pad to next power of two (if needed) ---

        if padded_dim > self.input_dim:
            pad_width = padded_dim - self.input_dim
            # Pad only along the last axis with zeros
            pad_config = [(0, 0)] * x.ndim
            pad_config[-1] = (0, pad_width)
            x_padded = jnp.pad(x, pad_config)
        else:
            x_padded = x

        # --- 2. Random sign flip (D * x_padded) ---
        # Broadcast diag over batch dims
        x_padded = x_padded * diag

        # --- 3. Fast Walsh–Hadamard Transform (H * D * x_padded) ---
        x_hadamard = fwht(x_padded)

        # --- 4. Subsample coordinates (S * H * D * x_padded) ---
        x_sub = jnp.take(x_hadamard, indices, axis=-1)

        # --- 5. Final scaling (1 / sqrt(k)) ---
        y = x_sub / jnp.sqrt(self.output_dim)

        return y
