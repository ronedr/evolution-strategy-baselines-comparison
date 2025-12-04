
import jax
import jax.numpy as jnp
import inspect
from evosax.algorithms.distribution_based.evotf_es import EvoTransformer

print("EvoTransformer found")

model = EvoTransformer(
    embed_dim=128,
    num_heads=1,
    num_latents=8,
    latent_dim=32,
    num_layers=1,
    use_fitness_encoder=True,
    use_dist_encoder=True,
    use_crossd_encoder=False
)

rng = jax.random.PRNGKey(0)

# Dummy inputs
batch = 1
seq_len = 10
dims = 5

# solution_features: (batch, seq_len, dims)
x = jnp.zeros((batch, seq_len, dims))
# fitness_features: (batch, seq_len, 1) - assuming it expects expanded dim
f = jnp.zeros((batch, seq_len, 1))
# dist_features: (batch, seq_len, dims) - assuming it expects mean? or mean+std?
# Let's try dims first.
d = jnp.zeros((batch, seq_len, dims))

try:
    print("Attempting init with (x, f, d)...")
    params = model.init(rng, x, f, d)
    print("Model initialized with (x, f, d)")
    
    out = model.apply(params, x, f, d)
    print("Output type:", type(out))
    if hasattr(out, 'shape'):
         print("Output shape:", out.shape)
    elif isinstance(out, tuple):
        print("Output tuple len:", len(out))
        for i, o in enumerate(out):
            if hasattr(o, 'shape'):
                print(f"Output {i} shape:", o.shape)
            else:
                print(f"Output {i}:", o)
                
except Exception as e:
    print(f"Failed with (x, f, d): {e}")

