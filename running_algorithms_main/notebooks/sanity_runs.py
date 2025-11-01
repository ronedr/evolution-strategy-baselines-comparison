import os
import sys

sys.path.append('../../')
import jax
import optax
from typing import List

from evosax.problems import CNN, TorchVisionProblem, identity_output_fn
from evosax.algorithms import algorithms
from evosax.core.fitness_shaping import standardize_fitness_shaping_fn

from tqdm import tqdm
from utils.problem_utils import get_problem_settings
from experiment.experiment import Experiment
from utils.problem_utils import get_problem_name
from evosax.problems import Problem

# from experiment.run_experiments import run_experiment_permutations

# jax.config.update('jax_default_matmul_precision', 'tensorfloat32')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_autotune_level=1'


NUM_GENERATIONS = 10
POPULATION_SIZE = 2
SEEDS = [0]
RESULT_DIR = "../../results"
PROBLEMS_TORCH_VISION = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"][:1]

running_es = {
    #     "PGPE": {
    #         "optimizer": optax.adam(learning_rate=0.02),
    #     },
    #     "ASEBO": {
    #         "optimizer": optax.adam(learning_rate=0.01),
    #         "fitness_shaping_fn": standardize_fitness_shaping_fn
    #     },
    #     "LES": {
    #         "optimizer": optax.adam(learning_rate=0.01)
    #     },
    #     "Open_ES": {
    #         "optimizer": optax.adam(learning_rate=0.05)
    #     },

    # "SNES": {},
    # "Sep_CMA_ES": {},
    "CMA_ES": {},
    # "LES": {},
    # "DES": {},
    # "EvoTF_ES": {},
}


def run_experiment_permutations(
        problems: List[Problem],
        op_name_to_params_mapping: dict,
        num_generations: int,
        population_size: int,
        result_dir: str,
        run_again_if_exist: bool = False,
        seeds: list[int] = None,
):
    if seeds is None:
        seeds = list(range(5))

    for problem in problems:
        for es_name in tqdm(op_name_to_params_mapping, desc="Running ES algorithms"):
            for seed in seeds:
                try:
                    # Create a master key for the current seed
                    key = jax.random.key(seed)
                    # Split a subkey for initializing the algorithm's solution
                    key, subkey_init = jax.random.split(key)
                    es_algorithm = algorithms[es_name](
                        population_size=population_size,
                        solution=problem.sample(subkey_init),
                        **op_name_to_params_mapping[es_name],
                    )
                    experiment = Experiment(
                        problem=problem,
                        algorithm=es_algorithm,
                        results_dir_path=result_dir,
                        seed=seed,
                        log_period=2,
                        eval_batch_size=2
                    )

                    if run_again_if_exist or not experiment.has_run():
                        print(f"Running experiment: {experiment.get_experiment_path_file()}")
                        experiment.run(num_generations)
                    else:
                        print(f"Experiment results already exist: {experiment.get_experiment_path_file()}")
                except Exception as e:
                    print(
                        f"Failed to run experiment for {es_name} on {get_problem_name(problem)} with seed {seed}. Error: {e}")


for task_name in tqdm(PROBLEMS_TORCH_VISION, desc="Loading Problems .."):
    try:
        current_task = TorchVisionProblem(task_name=task_name,
                                          network=CNN(
                                              num_filters=[4],
                                              kernel_sizes=[(5, 5)],
                                              strides=[(1, 1)],
                                              mlp_layer_sizes=[5],
                                              output_fn=identity_output_fn,
                                          ),
                                          batch_size=8)
        print("Successfully loaded:", task_name)
        run_experiment_permutations(problems=[current_task],
                                    op_name_to_params_mapping=running_es,
                                    num_generations=NUM_GENERATIONS,
                                    population_size=POPULATION_SIZE,
                                    result_dir=RESULT_DIR,
                                    run_again_if_exist=True,
                                    seeds=SEEDS)
    except Exception as e:
        print("Failed to load: " + task_name, '\nREASON:', e)
        raise e
