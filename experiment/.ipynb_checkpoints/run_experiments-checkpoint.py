from typing import List

import jax
from evosax.algorithms import algorithms
from evosax.problems import Problem
from evosax.problems.rl.brax import BraxProblem
from evosax.problems.rl.gymnax import GymnaxProblem

from experiment.compare_results import compare
from experiment.experiment import Experiment
from utils.problem_utils import get_problem_name
from tqdm import tqdm


def run_experiment_permutations(problems: List[Problem], es_dict: dict, num_generations: int, population_size: int,
                                seed: int, result_dir: str):
    for problem in problems:
        key = jax.random.key(seed)
        key, subkey = jax.random.split(key)
        solution = problem.sample(subkey)


        for es_name in tqdm(es_dict, desc="Running ES algorithms"):
            ES = algorithms[es_name]
            es = ES(population_size=population_size, solution=solution, **es_dict[es_name])
            minimize_fitness = isinstance(problem, GymnaxProblem) or isinstance(problem, BraxProblem)
            Experiment(key, problem, es, results_dir_path=result_dir, minimize_fitness=minimize_fitness).run(
                num_generations=num_generations)

        compare(results_dir_path=result_dir, folder_path_problem=get_problem_name(problem), y_graph="best_fitness",
                x_graph="generation_counter", algorithms="*")

        compare(results_dir_path=result_dir, folder_path_problem=get_problem_name(problem), y_graph="best_fitness",
                x_graph="gen_time_sec", algorithms="*")
