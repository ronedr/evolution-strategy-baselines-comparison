import jax.numpy as jnp
from flax import linen as nn

from deep_noise_modeling.projections.srht_random_projection import SRHT_Projection_Padded


class RandomProjectionSRHT(nn.Module):
    """
    Interface-compatible SRHT projection with stacked layers.

    Constructor is called as:
        RandomProjectionSRHT(input_dim=input_dim, output_dim=random_projection_dim)

    Args:
        input_dim: dimension of the input
        output_dim: final output dimension after concatenation
        n_layers: number of independent SRHT projections (default=4)

    Output:
        shape = [..., output_dim]
    """
    input_dim: int
    output_dim: int
    n_layers: int = 4  # default stacking factor

    @nn.compact
    def __call__(self, x):
        # Validate input dim
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Last dimension {x.shape[-1]} != input_dim {self.input_dim}"
            )

        # Ensure output_dim is divisible by n_layers
        if self.output_dim % self.n_layers != 0:
            raise ValueError(
                f"output_dim ({self.output_dim}) must be divisible by n_layers ({self.n_layers})."
            )

        # Compute hidden dim per SRHT layer
        layer_hidden_dim = self.output_dim // self.n_layers

        outputs = []
        for i in range(self.n_layers):
            srht_layer = SRHT_Projection_Padded(
                input_dim=self.input_dim,
                output_dim=layer_hidden_dim,
                name=f"srht_layer_{i}"
            )
            outputs.append(srht_layer(x))

        # Concatenate along last dimension → full output_dim
        return jnp.concatenate(outputs, axis=-1)
