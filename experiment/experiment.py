import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from evosax.algorithms.base import EvolutionaryAlgorithm
from evosax.problems import Problem, TorchVisionProblem
from evosax.problems.rl.brax import BraxProblem
from evosax.problems.rl.gymnax import GymnaxProblem

from utils.jax_utils import to_list
from utils.problem_utils import get_problem_name


class Experiment:
    METRIC_KEYS = ['best_fitness', 'best_fitness_in_generation', 'mean_fitness_in_generation',
                   'best_accuracy_in_generation', 'mean_accuracy_in_generation',
                   'mean_fitness_in_generation_test', 'mean_accuracy_in_generation_test',
                   'generation_counter', 'gen_time_sec']

    def __init__(self, problem: Problem, algorithm: EvolutionaryAlgorithm, results_dir_path: str, seed: int,
                 log_period=16):
        self._seed = seed
        self._problem = problem
        self._algorithm = algorithm
        self._results_dir_path = results_dir_path
        self._minimize_fitness = isinstance(problem, GymnaxProblem) or isinstance(problem, BraxProblem)
        self.log_period = log_period

    def run(self, num_generations: int):
        # Split off main RNG key
        key = jax.random.PRNGKey(self._seed)
        key, subkey = jax.random.split(key)

        # Initialize ES algorithm state
        params = self._algorithm.default_params
        state = self._algorithm.init(subkey, self._algorithm.solution, params)

        # Initialize problem-specific state
        key, subkey = jax.random.split(key)
        problem_state = self._problem.init(subkey)

        collected_metrics = []
        for i in range(num_generations // self.log_period):
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, self.log_period)
            (state, params, problem_state), metrics = jax.lax.scan(
                self._step,
                (state, params, problem_state),
                keys,
            )
            collected_metrics.append(metrics)
            if isinstance(self._problem, TorchVisionProblem):
                print(
                    f"Generation {(i + 1) * self.log_period:03d}"
                    f" | Mean fitness (Test): {metrics['mean_fitness_in_generation_test'][-1]:.2f}"
                    f" | Mean Accuracy (Test): {metrics['mean_accuracy_in_generation_test'][-1]:.2f}")
            else:
                print(
                    f"Generation {(i + 1) * self.log_period:03d}"
                    f" | Mean fitness (Test): {metrics['mean_fitness_in_generation_test'][-1]:.2f}")
        
        print(jax.__version__)
        # Optionally stack metrics if needed for downstream analysis
        metrics = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *collected_metrics)

        # Convert per-generation runtime to cumulative time for plotting/analysis
        metrics["cum_gen_time_sec"] = np.cumsum(metrics["gen_time_sec"])

        # Save or export results (fitness, runtime, algorithm-specific stats)
        self._save_result(metrics)
        if isinstance(self._problem, BraxProblem):
            try:
                self._write_gif_best_running_visualization(key, state, problem_state)
            except Exception as e:
                print(e)
        return metrics

    def _write_gif_best_running_visualization(self, key, state, problem_state):
        mean = self._algorithm._unravel_solution(state.best_solution)

        key, subkey = jax.random.split(key)
        fitness, problem_state, info = self._problem.eval(key, jax.tree.map(lambda x: x[None], mean), problem_state)

        if isinstance(self._problem, BraxProblem):
            rollout = [
                jax.tree_util.tree_map(lambda x: x[0, 0, t], info["env_states"].pipeline_state)
                for t in range(self._problem.episode_length)
            ]
            from brax.io import html
            html_content = html.render(
                self._problem.env.sys.tree_replace({"opt.timestep": self._problem.env.dt}), rollout
            )
            with open(f"{self.get_experiment_path_file()}.html", "w") as f:
                f.write(html_content)

    def get_experiment_path_file(self):
        folder_path = f"{self._results_dir_path}/{get_problem_name(self._problem)}/{self._algorithm.__class__.__name__}"
        os.makedirs(folder_path, exist_ok=True)
        return f"{folder_path}/{self._seed}"

    def _save_result(self, metrics):
        # Save to a .json file
        cpu_metrics = to_list(metrics)
        with open(f"{self.get_experiment_path_file()}.json", "w") as f:
            json.dump(cpu_metrics, f, indent=4)

    def _step(self, carry, key):
        state, params, problem_state = carry
        key_ask, key_eval, key_tell = jax.random.split(key, 3)

        start = time.time()

        # 1. Sample candidates = populations
        population, state = self._algorithm.ask(key_ask, state, params)

        # 2. Evaluate the candidates (handles Brax, Gymnax, BBOB, MNIST, etc.)
        # info using only if the proplem instance of vision problems.
        fitness, problem_state, info = self._problem.eval(key_eval, population, problem_state)

        # In Brax and Gymnax Problems we want to maximis the fitness.
        fitness = -fitness if self._minimize_fitness else fitness

        # 3. Update ES state and collect algorithm-specific metrics
        state, metrics = self._algorithm.tell(key_tell, population, fitness, state, params)

        # 4. Add custom metrics: mean fitness and runtime

        # - add accuracy metric on population on the train dataset.
        extra_metrics = {
            "best_accuracy_in_generation": info["accuracy"][jnp.argmax(info["accuracy"])],
            "mean_accuracy_in_generation": jnp.mean(info["accuracy"])
        } if "accuracy" in info else {}

        # - add runtime metric.
        metrics = {**metrics,
                   **extra_metrics,
                   "gen_time_sec": time.time() - start,
                   "mean_fitness_in_generation": jnp.mean(fitness)}

        # - evalualte the new mean.
        test_metrics = self._eval_test(key_eval, state, problem_state)

        # - update metrics and drop the irrelevant metrics.
        metrics = self._update_metrics(metrics, test_metrics)

        return (state, params, problem_state), metrics

    def _extract_metrics(self, accuracy, fitness, key_eval, key_tell, metrics, params, population, problem_state,
                         state):
        mean_solution = self._algorithm.get_mean(state)

        if isinstance(self._problem, TorchVisionProblem):
            fitness_test, _, accuracy_test = self._problem.eval_test(key_eval,
                                                                     jax.tree.map(lambda x: x[None], mean_solution),
                                                                     problem_state)
            _, test_metrics = self._algorithm.tell(key_tell, population, fitness_test, state, params)

            test_metrics = {f"{k}_test": v for k, v in test_metrics.items()}
            extra_metrics = {
                "mean_fitness_in_generation_test": jnp.mean(fitness_test),
                "mean_accuracy_in_generation_test": jnp.mean(accuracy_test["accuracy"]),

                "best_accuracy_in_generation": accuracy["accuracy"][jnp.argmax(accuracy["accuracy"])],
                "mean_accuracy_in_generation": jnp.mean(accuracy["accuracy"]),
            }
        metrics = self._update_metrics(metrics, extra_metrics)
        return metrics

    def _update_metrics(self, metrics, update_metrics):

        metrics = {**metrics, **update_metrics}
        metrics = {k: -metrics[k] if 'fitness' in k and self._minimize_fitness else metrics[k] for k in
                   Experiment.METRIC_KEYS if k in metrics}

        return metrics

    def has_run(self):
        return os.path.exists(f"{self.get_experiment_path_file()}.json")

    def _eval_test(self, key, state, problem_state):
        key, key_eval = jax.random.split(key)
        mean_solution = self._algorithm.get_mean(state)

        if isinstance(self._problem, TorchVisionProblem):
            fitness, _, info = self._problem.eval_test(key_eval,
                                                       jax.tree.map(lambda x: x[None], mean_solution),
                                                       problem_state)

            return {
                "mean_fitness_in_generation_test": jnp.mean(fitness),
                "mean_accuracy_in_generation_test": jnp.mean(info["accuracy"]),
            }

        else:
            fitness, _, info = self._problem.eval(key_eval,
                                                  jax.tree.map(lambda x: x[None], mean_solution),
                                                  problem_state)
            fitness = -fitness if self._minimize_fitness else fitness
            return {
                "mean_fitness_in_generation_test": jnp.mean(fitness)
            }
