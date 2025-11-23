import jax
import jax.numpy as jnp
from deep_noise_modeling.vae_unet import VAEUNetEncoder

def test_vae_unet():
    input_dim = 100
    latent_dim = 32
    batch_size = 4
    
    encoder = VAEUNetEncoder(input_dim=input_dim, latent_dim=latent_dim)
    
    key = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((batch_size, input_dim))
    
    variables = encoder.init(key, dummy_input)
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
