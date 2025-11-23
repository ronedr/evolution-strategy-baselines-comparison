import jax
import jax.numpy as jnp
from flax import linen as nn


class VAEUnetEncoder(nn.Module):
    input_dim: int
    latent_dim: int = 32
    base_filters: int = 16
    depth: int = 2
    activate: callable = nn.relu

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, input_dim)
        # Reshape to (batch, input_dim, 1) for 1D CNN
        h = x.reshape((x.shape[0], -1, 1))
        
        # Encoder
        skips = []
        for i in range(self.depth):
            # Double conv block
            h = nn.Conv(features=self.base_filters * (2 ** i), kernel_size=(3,), padding='SAME')(h)
            h = self.activate(h)
            h = nn.Conv(features=self.base_filters * (2 ** i), kernel_size=(3,), padding='SAME')(h)
            h = self.activate(h)
            
            skips.append(h)
            # Downsample
            h = nn.max_pool(h, window_shape=(2,), strides=(2,), padding='SAME')

        # Bottleneck
        h = nn.Conv(features=self.base_filters * (2 ** self.depth), kernel_size=(3,), padding='SAME')(h)
        h = self.activate(h)
        h = nn.Conv(features=self.base_filters * (2 ** self.depth), kernel_size=(3,), padding='SAME')(h)
        h = self.activate(h)

        # Flatten and concatenate all features (skips + bottleneck)
        # This provides a multi-scale dense representation
        flat_features = [h.reshape((h.shape[0], -1))]
        
        for skip in skips:
            flat_features.append(skip.reshape((skip.shape[0], -1)))
            
        h = jnp.concatenate(flat_features, axis=-1)

        # Dense layers for VAE
        mu = nn.Dense(self.latent_dim, name="fc_mu")(h)
        logvar = nn.Dense(self.latent_dim, name="fc_logvar")(h)
        std = jnp.exp(0.5 * logvar)
        
        return mu, std, logvar
