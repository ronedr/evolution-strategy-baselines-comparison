import jax
import jax.numpy as jnp
from count_sketch_projection import CountSketch_Projection

input_dim = 500_000
output_dim = 4096

proj = CountSketch_Projection(input_dim=input_dim, output_dim=output_dim)

key = jax.random.key(0)
x = jax.random.normal(key, (8, input_dim))  # batch=8

params = proj.init(key, x)
y = proj.apply(params, x)

print("Output shape:", y.shape)
print(y)
