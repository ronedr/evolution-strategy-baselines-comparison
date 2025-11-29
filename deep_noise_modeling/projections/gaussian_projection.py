import jax
import jax.numpy as jnp
from flax import linen as nn


class GaussianProjection(nn.Module):
    """
    Gaussian Random Projection:
        R_ij ~ Normal(0, 1/output_dim)
    Projection:
        y = x @ R

    Input:
        x: [..., input_dim]
    Output:
        y: [..., output_dim]

    The matrix R is created once during init() and is fixed afterward.
    """
    input_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Last dimension {x.shape[-1]} != expected input_dim {self.input_dim}"
            )

        # --- Gaussian RP matrix initialization ---
        def gaussian_init(key, shape):
            # Entries ~ N(0, 1/output_dim)
            std = 1.0 / jnp.sqrt(self.output_dim)
            return std * jax.random.normal(key, shape)

        # Shape = (input_dim, output_dim)
        R = self.param(
            "R", gaussian_init, (self.input_dim, self.output_dim)
        )

        # --- Dense projection ---
        # y = x @ R
        return x @ R
