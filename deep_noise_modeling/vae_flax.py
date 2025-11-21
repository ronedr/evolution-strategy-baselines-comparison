import jax.numpy as jnp
from flax import linen as nn


class VAEEncoder(nn.Module):
    input_dim: int
    latent_dim: int = 32
    hidden_dims: tuple = (128, 64)
    activate: callable = nn.relu

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, input_dim)
        h = x
        for hdim in self.hidden_dims:
            h = nn.Dense(hdim)(h)
            h = self.activate(h)

        mu = nn.Dense(self.latent_dim, name="fc_mu")(h)
        logvar = nn.Dense(self.latent_dim, name="fc_logvar")(h)
        std = jnp.exp(0.5 * logvar)
        return mu, std, logvar
