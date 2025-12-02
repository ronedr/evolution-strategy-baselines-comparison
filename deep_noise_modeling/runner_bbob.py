import jax
import jax.numpy as jnp
import optax

from evosax.problems.bbob.bbob_fns import bbob_fns
from vae_open_es import VAE_Open_ES as ES
# from evosax.algorithms import Open_ES as ES

# -------------------------
# 1. Choose BBOB function
# -------------------------
fn_name = "sphere"  # <-- change this to: rastrigin, rosenbrock, bent_cigar, etc.
bbob_fn = bbob_fns[fn_name]

num_dims = 10  # dimensionality of the challenge
POPULATION_SIZE = 6
num_generations = 100


# -------------------------
# 2. BBOB function wrapper
# -------------------------
def evaluate_population(pop):
    """Evaluate each individual in the population using the BBOB API."""
    # BBOB API always requires these parameters:
    # f(x, x_opt, R, Q, num_dims)
    # But for simple benchmarking, we can set identity transforms + zero opt.

    x_opt = jnp.zeros(num_dims)  # optimum at 0
    R = jnp.eye(num_dims)  # no rotation
    Q = jnp.eye(num_dims)  # no conditioning

    # bbob_fn returns (value, penalty)
    def eval_single(x):
        val, penalty = bbob_fn(x, x_opt, R, Q, num_dims)
        return val + penalty  # combine penalty into fitness

    return jax.vmap(eval_single)(pop)  # vectorized evaluation


# -------------------------
# 3. Initial solution
# -------------------------
key = jax.random.key(0)
key, subkey = jax.random.split(key)

solution = jax.random.uniform(subkey, (num_dims,), minval=-5.0, maxval=5.0)
print("Problem dimensionality:", num_dims)

# -------------------------
# 4. ES hyperparameters
# -------------------------

lr_schedule = optax.exponential_decay(
    init_value=0.05,
    transition_steps=num_generations,
    decay_rate=0.2,
)

std_schedule = optax.exponential_decay(
    init_value=0.1,
    transition_steps=num_generations,
    decay_rate=0.5,
)

# -------------------------
# 5. Instantiate your ES
# -------------------------
es = ES(
    population_size=POPULATION_SIZE,
    solution=solution,
    optimizer=optax.adam(learning_rate=lr_schedule),
    std_schedule=std_schedule,
    use_best_individual_augmentation=True
)

params = es.default_params


# -------------------------
# 6. One ES step
# -------------------------
def step(carry, key):
    state, params = carry

    key_ask, key_eval, key_tell = jax.random.split(key, 3)

    # ASK
    population, state = es.ask(key_ask, state, params)

    # Evaluate BBOB population
    fitness = evaluate_population(population)

    # TELL
    state, metrics = es.tell(key_tell, population, fitness, state, params)

    return (state, params), metrics


# -------------------------
# 7. Initialize state
# -------------------------
key, subkey = jax.random.split(key)
state = es.init(subkey, solution, params)

# -------------------------
# 8. Training loop
# -------------------------
for gen in range(num_generations):
    key, subkey = jax.random.split(key)
    (state, params), metrics = step((state, params), subkey)

    # Logging
    if (gen + 1) % 10 == 0:
        mean = es.get_mean(state)
        score = evaluate_population(mean[None])[0]
        print(f"Gen {gen + 1:03d} | Best-so-far fitness: {score:.6f}")
