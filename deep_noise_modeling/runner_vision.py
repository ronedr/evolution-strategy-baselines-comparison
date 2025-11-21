import jax
import optax
from evosax.problems import CNN, TorchVisionProblem as Problem, identity_output_fn
from vae_open_es import VAE_Open_ES as ES
from evosax.algorithms import EvoTF_ES

seed = 0
key = jax.random.key(seed)

network = CNN(
    num_filters=[8, 16],
    kernel_sizes=[(5, 5), (5, 5)],
    strides=[(1, 1), (1, 1)],
    mlp_layer_sizes=[10],
    output_fn=identity_output_fn,
)

problem = Problem(
    task_name="MNIST",
    network=network,
    batch_size=1024,
)

key, subkey = jax.random.split(key)
problem_state = problem.init(key)

key, subkey = jax.random.split(key)
solution = problem.sample(subkey)
print(f"Number of pararmeters: {sum(leaf.size for leaf in jax.tree.leaves(solution))}")

num_generations = 64
lr_schedule = optax.exponential_decay(
    init_value=0.01,
    transition_steps=num_generations,
    decay_rate=0.1,
)
std_schedule = optax.exponential_decay(
    init_value=0.05,
    transition_steps=num_generations,
    decay_rate=0.2,
)
es = ES(
    population_size=10,
    solution=solution,
    optimizer=optax.adam(learning_rate=lr_schedule),
    std_schedule=std_schedule,
)

params = es.default_params


def step(carry, key):
    state, params, problem_state = carry
    key_ask, key_eval, key_tell = jax.random.split(key, 3)

    population, state = es.ask(key_ask, state, params)

    fitness, problem_state, _ = problem.eval(key_eval, population, problem_state)

    state, metrics = es.tell(
        key_tell, population, fitness, state, params
    )  # Minimize fitness

    return (state, params, problem_state), metrics


key, subkey = jax.random.split(key)
state = es.init(subkey, solution, params)

fitness_log = []
log_period = 32
for i in range(num_generations // log_period):
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, log_period)
    (state, params, problem_state), metrics = jax.lax.scan(
        step,
        (state, params, problem_state),
        keys,
    )

    mean = es.get_mean(state)

    key, subkey = jax.random.split(key)
    fitness, problem_state, info = problem.eval_test(
        key, jax.tree.map(lambda x: x[None], mean), problem_state
    )
    print(
        f"Generation {(i + 1) * log_period:03d} | Mean fitness: {fitness.mean():.2f} | Accuracy: {info['accuracy'].mean():.2f}"
    )
