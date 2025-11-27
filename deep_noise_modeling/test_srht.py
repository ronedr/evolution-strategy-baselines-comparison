import jax
import jax.numpy as jnp
from srht_random_projection import SRHT_Projection_Padded

def test_srht():
    # Suppose your real input dim is 500_000 (not a power of 2)
    input_dim = 10_000
    output_dim = 4096
    BATCH_SIZE = 8

    proj = SRHT_Projection_Padded(input_dim=input_dim, output_dim=output_dim)

    key = jax.random.key(0)
    x = jax.random.normal(key, (BATCH_SIZE, input_dim))  # batch of 8

    # Initialize params (diag + indices) and apply
    print("Initializing...")
    params = proj.init(key, x)
    print("Applying...")
    y = proj.apply(params, x)

    print("Input shape:", x.shape)  # (8, 500000)
    print("Output shape:", y.shape)  # (8, 4096)
    
    # Verify shapes
    assert x.shape == (BATCH_SIZE, input_dim)
    assert y.shape == (BATCH_SIZE, output_dim)
    print("Test passed!")

if __name__ == "__main__":
    test_srht()
