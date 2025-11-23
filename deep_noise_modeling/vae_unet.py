import jax
import jax.numpy as jnp
from flax import linen as nn


class DownBlock(nn.Module):
    features: int
    kernel_size: int = 3
    stride: int = 2

    @nn.compact
    def __call__(self, x):
        # Conv1D with stride for downsampling
        x = nn.Conv(features=self.features, kernel_size=(self.kernel_size,), strides=(self.stride,))(x)
        x = nn.relu(x)
        return x


class UpBlock(nn.Module):
    features: int
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x, skip):
        # Upsample using nearest neighbor interpolation
        B, L, C = x.shape
        x = jax.image.resize(x, shape=(B, L * 2, C), method='nearest')
        
        # If the upsampled shape doesn't match skip connection (due to odd input size), crop or pad
        # For simplicity in this VAE context, we assume input dims are friendly or we handle simple mismatch
        # Let's align to skip shape
        if x.shape[1] != skip.shape[1]:
             x = x[:, :skip.shape[1], :]

        x = nn.Conv(features=self.features, kernel_size=(self.kernel_size,))(x)
        x = nn.relu(x)
        
        x = jnp.concatenate([x, skip], axis=-1)
        
        x = nn.Conv(features=self.features, kernel_size=(self.kernel_size,))(x)
        x = nn.relu(x)
        return x


class VAEUNetEncoder(nn.Module):
    input_dim: int
    latent_dim: int = 32
    # Channels for each level of the U-Net. 
    # Example: (32, 64, 128) means 3 levels of depth.
    channels: tuple = (32, 64, 128) 
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, input_dim)
        # Reshape to (batch, input_dim, 1) for Conv1D
        x = x[..., None]
        
        # Initial projection
        x = nn.Conv(features=self.channels[0], kernel_size=(self.kernel_size,))(x)
        x = nn.relu(x)
        
        skips = []
        
        # Downsampling path
        for features in self.channels[1:]:
            skips.append(x)
            x = DownBlock(features=features, kernel_size=self.kernel_size)(x)
            
        # Bottom bottleneck
        x = nn.Conv(features=self.channels[-1] * 2, kernel_size=(self.kernel_size,))(x)
        x = nn.relu(x)
        
        # Upsampling path
        # We iterate backwards through channels (excluding the last one which was used in bottleneck/last down)
        for i, features in enumerate(reversed(self.channels[:-1])):
            skip = skips.pop()
            x = UpBlock(features=features, kernel_size=self.kernel_size)(x, skip)
            
        # Final convolution to mix
        x = nn.Conv(features=self.channels[0], kernel_size=(self.kernel_size,))(x)
        x = nn.relu(x)
        
        # Flatten
        x = x.reshape((x.shape[0], -1))
        
        # Dense projection to latent space
        mu = nn.Dense(self.latent_dim, name="fc_mu")(x)
        logvar = nn.Dense(self.latent_dim, name="fc_logvar")(x)
        std = jnp.exp(0.5 * logvar)
        
        return mu, std, logvar
