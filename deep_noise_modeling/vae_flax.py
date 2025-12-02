import jax.numpy as jnp
from flax import linen as nn


class VAEEncoder(nn.Module):
    input_dim: int
    latent_dim: int = 32
    hidden_dims: tuple = (128, 64)
    activate: callable = nn.relu
    use_layernorm: bool = False
    use_dropout: bool = False
    dropout_rate: float = 0.1
    deterministic: bool = True  # important for dropout/batchnorm in apply()

    @nn.compact
    def __call__(self, x):
        h = x
        for i, hdim in enumerate(self.hidden_dims):
            h = nn.Dense(hdim)(h)

            if self.use_layernorm:
                h = nn.LayerNorm()(h)

            h = self.activate(h)

            if self.use_dropout:
                h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=self.deterministic)

        # Latent mean and logvar
        mu = nn.Dense(self.latent_dim, name="fc_mu")(h)
        logvar = nn.Dense(self.latent_dim, name="fc_logvar")(h)
        std = jnp.exp(0.5 * logvar)
        return mu, std, logvar
