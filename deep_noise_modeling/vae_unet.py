import jax
import jax.numpy as jnp
from flax import linen as nn


# ------------------------------------------------------------
# Downsampling Block (Conv1D + stride)
# ------------------------------------------------------------
class DownBlock(nn.Module):
    features: int
    kernel_size: int = 3
    stride: int = 2

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            padding='SAME'
        )(x)
        x = nn.relu(x)
        return x


# ------------------------------------------------------------
# Upsampling Block (resize + conv + skip)
# ------------------------------------------------------------
class UpBlock(nn.Module):
    features: int
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x, skip):
        B, L, C = x.shape

        # Nearest neighbor upsample ×2
        x_up = jax.image.resize(
            x,
            shape=(B, L * 2, C),
            method='nearest'
        )

        # Align to skip shape
        if x_up.shape[1] > skip.shape[1]:
            x_up = x_up[:, :skip.shape[1], :]
        elif x_up.shape[1] < skip.shape[1]:
            pad = skip.shape[1] - x_up.shape[1]
            x_up = jnp.pad(x_up, ((0,0),(0,pad),(0,0)))

        # Conv + activation
        x_up = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            padding='SAME'
        )(x_up)
        x_up = nn.relu(x_up)

        # Skip concat
        x_cat = jnp.concatenate([x_up, skip], axis=-1)

        # Refinement conv
        x_cat = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            padding='SAME'
        )(x_cat)
        x_cat = nn.relu(x_cat)

        return x_cat


# ------------------------------------------------------------
# Spatial UNet VAE Encoder with dual decoders
# ------------------------------------------------------------
class VAEUNetEncoder(nn.Module):
    input_dim: int                  # raw sequence length
    latent_dim: int                 # desired latent sequence length
    channels: tuple = (32, 64, 128)
    latent_channels: int = 8        # channels in latent feature map
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x):

        # --------------------------------------------------------
        # (B, L) → (B, L, 1)
        # --------------------------------------------------------
        x = x[..., None]
        skips = []

        # Initial conv
        h = nn.Conv(
            features=self.channels[0],
            kernel_size=(self.kernel_size,),
            padding='SAME'
        )(x)
        h = nn.relu(h)

        # ---------------- ENCODER ----------------
        for ch in self.channels[1:]:
            skips.append(h)
            h = DownBlock(ch, kernel_size=self.kernel_size)(h)

        # Bottleneck
        bottleneck = nn.Conv(
            features=self.channels[-1],
            kernel_size=(self.kernel_size,),
            padding='SAME'
        )(h)
        bottleneck = nn.relu(bottleneck)

        # Reverse for decoding order
        rev_skips = list(reversed(skips))

        # ========================================================
        #  MU DECODER: upsample until reaching latent_dim
        # ========================================================
        mu_h = bottleneck
        target_len = self.latent_dim

        # Use UNet-like blocks but stop counting once exceeding latent_dim
        for i, ch in enumerate(reversed(self.channels[:-1])):
            mu_h = UpBlock(ch, kernel_size=self.kernel_size)(mu_h, rev_skips[i])
            if mu_h.shape[1] >= target_len:
                break

        # Resize final sequence EXACTLY to latent_dim
        mu_h = jax.image.resize(
            mu_h,
            shape=(mu_h.shape[0], target_len, mu_h.shape[-1]),
            method='nearest'
        )

        mu_map = nn.Conv(
            features=self.latent_channels,
            kernel_size=(1,),
            padding='SAME',
            name="mu_head"
        )(mu_h)

        # ========================================================
        #  LOGVAR DECODER BRANCH
        # ========================================================
        lv_h = bottleneck
        for i, ch in enumerate(reversed(self.channels[:-1])):
            lv_h = UpBlock(ch, kernel_size=self.kernel_size)(lv_h, rev_skips[i])
            if lv_h.shape[1] >= target_len:
                break

        lv_h = jax.image.resize(
            lv_h,
            shape=(lv_h.shape[0], target_len, lv_h.shape[-1]),
            method='nearest'
        )

        logvar_map = nn.Conv(
            features=self.latent_channels,
            kernel_size=(1,),
            padding='SAME',
            name="logvar_head"
        )(lv_h)

        # ========================================================
        #  SAMPLING: spatial VAE
        # ========================================================
        std_map = jnp.exp(0.5 * logvar_map)
        mu = jnp.mean(mu_map, axis=-1)  # pool channels → (B, L)
        std = jnp.mean(std_map, axis=-1)
        logvar = jnp.mean(logvar_map, axis=-1)
        return mu, std, logvar
