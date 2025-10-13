## add to sys the main folder.
import sys
sys.path.append('/home/ronedr/evolution-strategy-baselines-comparison')

## Extract the es_algorithms that we want to run in this job.
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--es_algorithms', nargs='+', help='List of es_algorithms that we want to run in this job', required=True)
args = parser.parse_args()

## imports.
import jax
import optax
from tqdm import tqdm
import gymnax
from evosax.problems import GymnaxProblem as Problem
from utils.nn_model import CNN
from utils.problem_utils import get_problem_settings
from experiment.run_experiments import run_experiment_permutations
from evosax.core.fitness_shaping import standardize_fitness_shaping_fn

num_generations = 5000
population_size = 256
seeds = list(range(0, 5))
result_dir = "../../results"
problems_gymnax = ['Asterix-MinAtar', 'Breakout-MinAtar', 'Freeway-MinAtar', 'SpaceInvaders-MinAtar']

# all algorithms that we want to comapre with best params according the article "DISCOVERING EVOLUTION STRATEGIES VIA META-BLACK-BOX OPTIMIZATION".
es_dict = {
    "PGPE": {
        "optimizer": optax.adam(learning_rate=0.02),
    },
    "ASEBO": {
        "optimizer": optax.adam(learning_rate=0.01),
        "fitness_shaping_fn": standardize_fitness_shaping_fn
    },
    "LES": {
        "optimizer": optax.adam(learning_rate=0.01)
    },
    "Open_ES": {    
        "optimizer": optax.adam(learning_rate=0.05)
    },
    "SNES": {},
    "Sep_CMA_ES": {},
    "CMA_ES": {},
    "LES": {},
    "DES": {},
    # "EvoTF_ES": {},
}

# take only the es_algorithms we insert as arg.
running_es = {es: es_dict[es] for es in args.es_algorithms if es in es_dict}

from IPython.display import clear_output

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
        run_experiment_permutations(problems=[problem],
                                    es_dict=running_es,
                                    num_generations=num_generations,
                                    population_size=population_size,
                                    result_dir=result_dir, 
                                    run_again_if_exist=False,
                                    seeds=list(range(0, 5)))
    except Exception as e:
        print("Failed to load:", env_name, e)
        continue