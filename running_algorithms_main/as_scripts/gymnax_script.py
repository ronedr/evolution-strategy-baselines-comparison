import sys

sys.path.append('/home/ronedr/evolution-strategy-baselines-comparison')

# === imports (parent) ===
import optax
from tqdm import tqdm
from evosax.problems import GymnaxProblem as Problem
from experiment.utils.nn_model import CNN
from experiment.utils.problem_utils import get_problem_settings
from experiment.run_experiments import run_experiment_permutations

# running params
num_generations = 5000
population_size = 256
eval_batch_size = 128
log_period = 10
seeds = list(range(0, 5))
result_dir = "../../results"
problems_gymnax = ['Asterix-MinAtar', 'Breakout-MinAtar', 'Freeway-MinAtar', 'SpaceInvaders-MinAtar']



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

# algorithms
es_dict = {
    "PGPE": {
        "optimizer": optax.adam(learning_rate=0.02),
    },
    "Open_ES": {    
        "optimizer": optax.adam(learning_rate=lr_schedule), 
        "std_schedule": std_schedule
    },
    "SNES": {},
    "Sep_CMA_ES": {},
    # "CMA_ES": {},
    "DES": {},
    "LES": {},
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
            suffix_experiment_name=f"{population_size}",
            seeds=seeds
        )
    except Exception as e:
        print("Failed to run:", env_name, e)
