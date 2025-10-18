# === Runtime safety flags MUST be set before importing jax ===
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85"
# allow fallback conv algos and lower autotune aggressiveness
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                           " --xla_gpu_strict_conv_algorithm_picker=false"
                           " --xla_gpu_autotune_level=1")
# optional, can help avoid flaky autotune paths
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# === add to sys the main folder. ===
import sys
sys.path.append('/home/ronedr/evolution-strategy-baselines-comparison')

## Extract the es_algorithms that we want to run in this job.
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--es_algorithms', help="List of es_algorithms that we want to run in this job, example: 'PGPE,Open_ES,ASEBO'", required=True)
args = parser.parse_args()
es_algorithms = args.es_algorithms.split(",")

## imports.
import gc
import jax
import optax
from tqdm import tqdm
import brax.envs as brax_envs
from evosax.problems import BraxProblem as Problem
from evosax.problems.networks import MLP
from utils.problem_utils import get_problem_settings
from experiment.run_experiments import run_experiment_permutations
from evosax.core.fitness_shaping import standardize_fitness_shaping_fn


# running params.
num_generations = 2000
population_size = 256
result_dir = "../../results" # back two steps need to change 
problems_brax_envs = list(brax_envs._envs.keys())


# all algorithms that we want to comapre with best params according the article "DISCOVERING EVOLUTION STRATEGIES VIA META-BLACK-BOX OPTIMIZATION".
es_dict = {
    "PGPE": {
        "optimizer": optax.adam(learning_rate=0.02),
    },
    "ASEBO": {
        "optimizer": optax.adam(learning_rate=0.01)
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
    "DES": {},
    # "EvoTF_ES": {},
}

# take only the es_algorithms we insert as arg.
running_es = {es: es_dict[es] for es in es_algorithms if es in es_dict}

for env_name in tqdm(problems_brax_envs, desc="Loading Problems .."):
    action_num, out_fn = get_problem_settings(env_name)
    try:
        problem = Problem(
            num_rollouts=1,
            env_name=env_name,
            policy=MLP(
                layer_sizes=(32, 32, 32, 32, action_num),
                output_fn=out_fn,
            ),
            episode_length=1000,
            env_kwargs={"backend": "generalized"},
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
    finally:
        # aggressively drop references and compiled executables between tasks
        try:
            del problem
        except NameError:
            pass
        jax.clear_caches()
        gc.collect()
