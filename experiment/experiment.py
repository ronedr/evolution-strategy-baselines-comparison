import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from evosax.algorithms.base import EvolutionaryAlgorithm
from evosax.problems import Problem
from gymnasium import spaces

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
        # Split off main RNG key
        key, subkey = jax.random.split(self._key)

        # Initialize ES algorithm state
        state = self._algorithm.init(subkey, self._algorithm.solution, self._algorithm.default_params)

        # Initialize problem-specific state
        key, subkey = jax.random.split(key)
        problem_state = self._problem.init(subkey)

        # Pre-generate RNG keys for each generation
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, num_generations)

        # Initialize carry
        carry = (state, self._algorithm.default_params, problem_state)
        # Prepare metrics list
        collected_metrics = []
        # Iterate over keys manually
        for key in keys:
            (state, params, problem_state), metrics = self._step(carry, key)
            collected_metrics.append(metrics)
            carry = (state, params, problem_state)
        # Optionally stack metrics if needed for downstream analysis
        metrics = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *collected_metrics)

        # Convert per-generation runtime to cumulative time for plotting/analysis
        metrics["cum_gen_time_sec"] = np.cumsum(metrics["gen_time_sec"])

        # Save or export results (fitness, runtime, algorithm-specific stats)
        self._save_result(metrics)
        return metrics

    def _save_result(self, metrics):
        # Save to a .json file
        folder_path = f"{self._results_dir_path}/{get_problem_name(self._problem)}"
        os.makedirs(folder_path, exist_ok=True)

        cpu_metrics = to_list(metrics)
        with open(f"{folder_path}/{self._algorithm.__class__.__name__}.json", "w") as f:
            json.dump(cpu_metrics, f, indent=4)

    def _step(self, carry, key):
        state, params, problem_state = carry
        key_ask, key_eval, key_tell = jax.random.split(key, 3)

        start = time.time()

        # 1. Sample candidate population
        population, state = self._algorithm.ask(key_ask, state, params)

        # 2. Clip to action bounds if applicable
        action_space = getattr(self._problem, "action_space", None)
        if isinstance(action_space, spaces.Box):
            population = jnp.clip(population, action_space.low, action_space.high)

        # 3. Evaluate candidates (handles Brax, RL, BBOB, MNIST, etc.)
        fitness, problem_state, _ = self._problem.eval(key_eval, population, problem_state)
        if self._minimize_fitness:
            fitness = -fitness

        # 4. Update ES state and collect algorithm-specific metrics
        state, metrics = self._algorithm.tell(key_tell, population, fitness, state, params)

        # 5. Add custom metrics: mean fitness and runtime
        metrics = self._update_metrics(metrics, {
            "mean_fitness": float(fitness.mean()),
            "gen_time_sec": time.time() - start
        })

        return (state, params, problem_state), metrics

    def _update_metrics(self, metrics, update_metrics):
        for metric in list(metrics.keys()):
            if "solution" in metric or "norm" in metric or metric == "mean":
                del metrics[metric]
            elif metric != "generation_counter" and metric != "gen_time_sec":
                metrics[metric] = -metrics[metric] if self._minimize_fitness else metrics[metric]

        return {**metrics, **update_metrics}
