import json
import os
import time

import jax
import numpy as np
from evosax.algorithms.base import EvolutionaryAlgorithm
from evosax.problems import Problem

from utils.jax_utils import to_list
from utils.problem_utils import get_problem_name


class Experiment:
    def __init__(self, key: float, problem: Problem, algorithm: EvolutionaryAlgorithm, results_dir_path: str,
                 print_progress: bool = False, minimize_fitness=False):
        self._key = key
        self._problem = problem
        self._algorithm = algorithm
        self._results_dir_path = results_dir_path
        self._print_progress = print_progress
        self._minimize_fitness = minimize_fitness

    def run(self, num_generations: int):
        key, subkey = jax.random.split(self._key)
        state = self._algorithm.init(subkey, self._algorithm.solution, self._algorithm.default_params)

        key, subkey = jax.random.split(key)
        problem_state = self._problem.init(subkey)

        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, num_generations)

        _, metrics = jax.lax.scan(self._step, (state, self._algorithm.default_params, problem_state), keys)
        metrics["gen_time_sec"] = np.cumsum(metrics["gen_time_sec"])
        self._save_result(metrics)
        return metrics

    def _save_result(self, metrics):
        # Save to a .pickle file
        folder_path = f"{self._results_dir_path}/{get_problem_name(self._problem)}"
        os.makedirs(folder_path, exist_ok=True)

        cpu_metrics = to_list(metrics)
        with open(f"{folder_path}/{self._algorithm.__class__.__name__}.json", "w") as f:
            json.dump(cpu_metrics, f, indent=4)

    def _step(self, carry, key):
        state, params, problem_state = carry
        key_ask, key_eval, key_tell = jax.random.split(key, 3)

        start_time = time.time()
        population, state = self._algorithm.ask(key_ask, state, params)
        fitness, problem_state, _ = self._problem.eval(key_eval, population, problem_state)
        fitness = -fitness if self._minimize_fitness else fitness
        state, metrics = self._algorithm.tell(key_tell, population, fitness, state, params)

        end_time = time.time()
        runtime = end_time - start_time
        metrics = self._update_metrics(metrics, {"gen_time_sec": runtime, "mean_fitness_in_generation": fitness.mean()})

        # if self._print_progress and metrics["generation_counter"][-1] % 50 == 0:
        #     print(
        #         f"Generation {metrics['generation_counter']} | Mean fitness: {fitness.mean():.2f}")

        return (state, params, problem_state), metrics

    def _update_metrics(self, metrics, update_metrics):
        for metric in list(metrics.keys()):
            if "solution" in metric or "norm" in metric or metric == "mean":
                del metrics[metric]
            elif metric != "generation_counter" and metric != "gen_time_sec":
                metrics[metric] = -metrics[metric] if self._minimize_fitness else metrics[metric]

        return {**metrics, **update_metrics}
