import jax
import jax.numpy as jnp
from deep_noise_modeling.vae_open_es import VAE_Open_ES

def test_top_k_augmentation():
    input_dim = 10
    pop_size = 20
    k = 5
    
    # Initialize strategy
    strategy = VAE_Open_ES(
        population_size=pop_size,
        solution=jnp.zeros(input_dim),
        k=k,
        use_best_individual_augmentation=True
    )
    
    # Init state
    rng = jax.random.PRNGKey(0)
    params = strategy.default_params
    state = strategy.init(rng, jnp.zeros(input_dim), params)
    
    # Check initial buffer
    assert state.top_k_solutions.shape == (k, input_dim)
    assert state.top_k_fitness.shape == (k,)
    assert jnp.all(jnp.isinf(state.top_k_fitness))
    
    # Create fake population and fitness
    # Fitness: 0 to 19
    population = jnp.zeros((pop_size, input_dim))
    # make population distinguishable
    population = population + jnp.arange(pop_size)[:, None] 
    fitness = jnp.arange(pop_size, dtype=jnp.float32)
    
    # Run ask to initialize noise state in state
    rng, key = jax.random.split(rng)
    _, state = strategy.ask(key, state, params)
    
    # Run tell
    rng, key = jax.random.split(rng)
    # Run tell
    rng, key = jax.random.split(rng)
    res = strategy.tell(key, population, fitness, state, params)
    if isinstance(res, tuple):
        state = res[0]
    else:
        state = res
    
    # Check if top-k buffer has the best 5 (0, 1, 2, 3, 4)
    print("Top k fitness:", state.top_k_fitness)
    assert jnp.all(jnp.sort(state.top_k_fitness) == jnp.arange(k, dtype=jnp.float32))
    
    # Run tell again with worse fitness
    worse_fitness = fitness + 100
    res = strategy.tell(key, population, worse_fitness, state, params)
    if isinstance(res, tuple):
        state = res[0]
    else:
        state = res
    
    # Top k should remain the same
    print("Top k fitness after worse update:", state.top_k_fitness)
    assert jnp.all(jnp.sort(state.top_k_fitness) == jnp.arange(k, dtype=jnp.float32))
    
    # Run tell with better fitness
    better_fitness = fitness - 10 # -10, -9, ...
    res = strategy.tell(key, population, better_fitness, state, params)
    if isinstance(res, tuple):
        state = res[0]
    else:
        state = res
    
    # Top k should be the new best (-10, -9, -8, -7, -6)
    expected_best = jnp.arange(k, dtype=jnp.float32) - 10
    print("Top k fitness after better update:", state.top_k_fitness)
    assert jnp.all(jnp.sort(state.top_k_fitness) == expected_best)
    
    # Test augmentation sampling
    # We will sample multiple times and ensure we get values from the top-k buffer
    rng, key = jax.random.split(rng)
    
    # Let's just check if we can run tell without error, which calls it.
    res = strategy.tell(key, population, fitness, state, params)
    if isinstance(res, tuple):
        state = res[0]
    else:
        state = res
    print("Tell with augmentation ran successfully.")

if __name__ == "__main__":
    test_top_k_augmentation()
    print("All tests passed!")
