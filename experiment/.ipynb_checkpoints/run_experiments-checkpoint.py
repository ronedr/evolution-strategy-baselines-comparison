from typing import List

import jax
from evosax.algorithms import algorithms
from evosax.problems import Problem
from tqdm import tqdm

from experiment.experiment import Experiment
from utils.problem_utils import get_problem_name


def run_experiment_permutations(problems: List[Problem], es_dict: dict, num_generations: int, population_size: int,
                                result_dir: str, run_again_if_exist: bool = False, seeds: list[int] = None,
                                log_period: int = 10, eval_batch_size=None):
    if seeds is None:
        seeds = list(range(0, 5))

    for problem in problems:
        for es_name in tqdm(es_dict, desc="Running ES algorithms"):
            try:
                for seed in seeds:
                    key = jax.random.PRNGKey(seed)  # was jax.random.key(...)
                    key, subkey = jax.random.split(key)
                    solution = problem.sample(subkey)

                    es_algorithm = algorithms[es_name](
                        population_size=population_size,
                        solution=solution,
                        **es_dict[es_name],
                    )

                    experiment = Experiment(
                        problem=problem,
                        algorithm=es_algorithm,
                        results_dir_path=result_dir,
                        log_period=log_period,
                        eval_batch_size=eval_batch_size,
                        seed=seed,
                    )

                    path = experiment.get_experiment_path_file()  # or experiment.metrics.experiment_path_file()
                    if not experiment.has_run() or run_again_if_exist:
                        print(f"running the experiment ... [{path}]")
                        experiment.run(num_generations=num_generations)
                    else:
                        print(f"there is experiment results. [{path}]")
            except Exception as e:
                print(e, es_name, get_problem_name(problem))
