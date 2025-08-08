from typing import List

import jax
from evosax.algorithms import algorithms
from evosax.problems import Problem
from evosax.problems.rl.brax import BraxProblem
from evosax.problems.rl.gymnax import GymnaxProblem
from tqdm import tqdm

from experiment.experiment import Experiment


def run_experiment_permutations(problems: List[Problem], es_dict: dict, num_generations: int, population_size: int,
                                seed: int, result_dir: str, run_again_if_exist: bool = False):
    for problem in problems:
        key = jax.random.key(seed)
        key, subkey = jax.random.split(key)
        solution = problem.sample(subkey)
        for es_name in tqdm(es_dict, desc="Running ES algorithms"):
            es_algorithm = algorithms[es_name](population_size=population_size,
                                               solution=solution,
                                               **es_dict[es_name])
            experiment = Experiment(problem=problem,
                                    algorithm=es_algorithm,
                                    results_dir_path=result_dir,
                                    minimize_fitness=isinstance(problem, GymnaxProblem) or isinstance(problem,
                                                                                                      BraxProblem),
                                    seed=seed)

            if not experiment.has_run() or run_again_if_exist:
                print(f"running the experiment ... [{experiment.get_experiment_path_file()}]")
                experiment.run(num_generations=num_generations)
            else:
                print(f"there is experiment results. [{experiment.get_experiment_path_file()}]")
