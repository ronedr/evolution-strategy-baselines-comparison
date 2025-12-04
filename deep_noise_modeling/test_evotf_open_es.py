
import jax
import jax.numpy as jnp
from deep_noise_modeling.evotf_open_es import EvoTF_Open_ES

import time

def run_test():
    print("Initializing EvoTF_Open_ES test...")
    
    # Define a simple problem
    num_dims = 5
    pop_size = 20
    
    # Instantiate algorithm
    # We need a dummy solution for init, usually just shape
    dummy_solution = jnp.zeros(num_dims)
    
    strategy = EvoTF_Open_ES(
        population_size=pop_size,
        solution=dummy_solution,
        num_dims=num_dims,
        context_len=5,
        use_antithetic_sampling=True,
        model_config={
            "embed_dim": 16,
            "num_heads": 2,
            "num_layers": 1,
            "latent_dim": 16
        },
        ppo_epochs=2
    )
    
    # Initialize
    rng = jax.random.PRNGKey(0)
    params = strategy.default_params
    state = strategy.init(rng, dummy_solution, params)
    
    print("Initialization successful.")
    print("State keys:", state.__dict__.keys())
    
    # Define a simple fitness function (Sphere)
    def fitness_fn(x):
        return jnp.sum(x**2)
    
    # Run loop
    print("\nStarting optimization loop...")
    for gen in range(10):
        rng, rng_ask, rng_tell = jax.random.split(rng, 3)
        
        # Ask
        x, state = strategy.ask(rng_ask, state, params)
        
        # Evaluate
        fitness = jax.vmap(fitness_fn)(x)
        
        # Tell
        state, metrics = strategy.tell(rng_tell, x, fitness, state, params)
        
        print(f"Gen {gen}: Best Fitness = {state.best_fitness:.4f}, Mean Fitness = {fitness.mean():.4f}")
        
    print("\nTest completed successfully.")

if __name__ == "__main__":
    run_test()
