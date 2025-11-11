import json
import os
from typing import Dict, Any

import jax
import jax.numpy as jnp
from evosax.algorithms.distribution_based.base import DistributionBasedAlgorithm
from evosax.problems import Problem
from evosax.problems.rl.brax import BraxProblem
from evosax.problems.rl.gymnax import GymnaxProblem

from experiment.utils.dict_utils import update_params
from experiment.utils.es_utils import scan_step
from experiment.utils.jax_utils import to_list
from experiment.utils.merics_utils import print_status, add_cumulative_time
from experiment.utils.problem_utils import get_problem_name
from experiment.utils.visualisation_utils import write_gif_best_running_visualization


class Experiment:
    def __init__(
            self,
            problem: Problem,
            algorithm: DistributionBasedAlgorithm,
            results_dir_path: str,
            seed: int,
            log_period: int,
            eval_batch_size: int | None = None,
    ):
        self._seed = seed
        self._problem = problem
        self._algorithm = algorithm
        self._results_dir_path = results_dir_path
        self._minimize_fitness = isinstance(problem, GymnaxProblem) or isinstance(problem, BraxProblem)
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else self._algorithm.population_size
        self.log_period = log_period

    def run(self, num_generations: int) -> Dict[str, Any]:
        key = jax.random.PRNGKey(self._seed)
        key, subkey = jax.random.split(key)

        params = self._algorithm.default_params
        params = update_params(params, {"std_init": 0.1})

        state = self._algorithm.init(subkey, self._algorithm.solution, params)

        key, subkey = jax.random.split(key)
        problem_state = self._problem.init(subkey)

        collected = []
        carry = (state, params, problem_state)

        for start in range(0, num_generations, self.log_period):
            chunk = min(self.log_period, num_generations - start)
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, chunk)

            (state, params, problem_state), period_gens_metrics = scan_step(
                self._algorithm, self._problem, self._minimize_fitness, carry, keys
            )
            carry = (state, params, problem_state)

            print_status(self._problem, chunk + start, period_gens_metrics)
            collected.append(period_gens_metrics)

        metrics = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *collected)
        metrics = add_cumulative_time(metrics)

        self.save_result(metrics)
        if isinstance(self._problem, BraxProblem):
            write_gif_best_running_visualization(self._algorithm, self._problem, self.get_experiment_path_file(), key,
                                                 state, problem_state)

        return metrics

    def has_run(self) -> bool:
        return os.path.exists(f"{self.get_experiment_path_file()}.json")

    def get_experiment_path_file(self) -> str:
        folder_path = f"{self._results_dir_path}/{get_problem_name(self._problem)}/{self._algorithm.__class__.__name__}"
        os.makedirs(folder_path, exist_ok=True)
        return f"{folder_path}/{self._seed}"

    def save_result(self, metrics: Dict[str, Any]) -> None:
        cpu_metrics = to_list(metrics)
        with open(f"{self.get_experiment_path_file()}.json", "w") as f:
            json.dump(cpu_metrics, f, indent=4)
