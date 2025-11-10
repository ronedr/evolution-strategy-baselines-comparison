import sys
sys.path.append('/home/ronedr/evolution-strategy-baselines-comparison')

# === imports (parent) ===
import jax
import gc
import multiprocessing as mp
import optax
from tqdm import tqdm
import gymnax
from evosax.problems import GymnaxProblem as Problem
from utils.nn_model import CNN
from utils.problem_utils import get_problem_settings
from experiment.run_experiments import run_experiment_permutations
from evosax.core.fitness_shaping import standardize_fitness_shaping_fn

# running params
num_generations = 5000
population_size = 256
eval_batch_size = 128
log_period = 10
seeds = list(range(0, 5))
result_dir = "../../results"
problems_gymnax = ['Asterix-MinAtar', 'Breakout-MinAtar', 'Freeway-MinAtar', 'SpaceInvaders-MinAtar']

# algorithms
es_dict = {
    "PGPE": {"optimizer": optax.adam(learning_rate=0.02)},
    "LES": {},
    "Open_ES": {
        "optimizer": optax.adam(learning_rate=0.05)
    },
    "SNES": {},
    "Sep_CMA_ES": {},
    "CMA_ES": {},
    "DES": {},
}


# === args ===
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--es_algorithms', required=True,
                    help="List of es_algorithms to run, e.g. 'PGPE,Open_ES,ASEBO'")
args = parser.parse_args()
es_algorithms = args.es_algorithms.split(",")

# filter by requested algos
running_es = {es: es_dict[es] for es in es_algorithms if es in es_dict}

for env_name in tqdm(problems_gymnax, desc="Loading Problems .."):
    try:
        action_num, out_fn = get_problem_settings(env_name)
        problem = Problem(
            env_name=env_name,
            policy=CNN(
                num_filters=[16],
                kernel_sizes=[(5, 5)],
                strides=(1, 1),
                mlp_layer_sizes=[32, action_num]
            ),
            num_rollouts=1,
            episode_length=500
        )
        print("Successfully loaded:", env_name)

        run_experiment_permutations(
            problems=[problem],
            es_dict=running_es,
            num_generations=num_generations,
            population_size=population_size,
            result_dir=result_dir,
            log_period=log_period,
            eval_batch_size=eval_batch_size,
            run_again_if_exist=False,
            seeds=seeds
        )
    except Exception as e:
        print("Failed to run:", env_name, e)
