import jax
import jax.numpy as jnp
from deep_noise_modeling.vae_unet import VAEUNetEncoder


def test_vae_unet():
    input_dim = 5000
    latent_dim = 5000
    batch_size = 1

    encoder = VAEUNetEncoder(input_dim=input_dim, latent_dim=latent_dim)
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((batch_size, input_dim))

    variables = encoder.init(key, dummy_input)
    num_params = sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: x.size, variables['params'])))
    print(f"Number of parameters: {num_params}")
    mu, std, logvar = encoder.apply(variables, dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Std shape: {std.shape}")
    print(f"Logvar shape: {logvar.shape}")

    assert mu.shape == (batch_size, latent_dim)
    assert std.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)
    print("Test passed!")


if __name__ == "__main__":
    test_vae_unet()
