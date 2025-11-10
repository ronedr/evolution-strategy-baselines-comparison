import json
import os
import time
from typing import Dict, Any

import jax.numpy as jnp
import numpy as np
from evosax.algorithms.base import EvolutionaryAlgorithm
from evosax.problems import Problem, TorchVisionProblem

from utils.jax_utils import to_list
from utils.problem_utils import get_problem_name


class MetricsService:
    METRIC_KEYS = [
        'best_fitness',
        'best_fitness_in_generation',
        'mean_fitness_in_generation',
        'best_accuracy_in_generation',
        'mean_accuracy_in_generation',
        'mean_fitness_in_generation_test',
        'mean_accuracy_in_generation_test',
        'generation_counter',
        'gen_time_sec',
    ]

    def __init__(
        self,
        results_dir_path: str,
        problem: Problem,
        algorithm: EvolutionaryAlgorithm,
        seed: int,
        minimize_fitness: bool,
    ):
        self.results_dir_path = results_dir_path
        self.problem = problem
        self.algorithm = algorithm
        self.seed = seed
        self.minimize_fitness = minimize_fitness

    # persistence
    def experiment_path_file(self) -> str:
        folder_path = f"{self.results_dir_path}/{get_problem_name(self.problem)}/{self.algorithm.__class__.__name__}"
        os.makedirs(folder_path, exist_ok=True)
        return f"{folder_path}/{self.seed}"

    def has_run(self) -> bool:
        return os.path.exists(f"{self.experiment_path_file()}.json")

    def save_result(self, metrics: Dict[str, Any]) -> None:
        cpu_metrics = to_list(metrics)
        with open(f"{self.experiment_path_file()}.json", "w") as f:
            json.dump(cpu_metrics, f, indent=4)

    # reporting
    def print_status(self, generation_count: int, metrics: Dict[str, Any]) -> None:
        print(
            f"Generation {generation_count:03d}"
            f" | Best fitness (Train): {float(metrics['best_fitness'][-1]):.2f}"
            f" | Best fitness in generation (Train): {float(metrics['best_fitness_in_generation'][-1]):.2f}"
            f" | Mean fitness (Train): {float(metrics['mean_fitness_in_generation'][-1]):.2f}"
            f" | Mean fitness (Test): {float(metrics['mean_fitness_in_generation_test'][-1]):.2f}"
        )
        if isinstance(self.problem, TorchVisionProblem):
            print(f" | Mean Accuracy (Test): {float(metrics['mean_accuracy_in_generation_test'][-1]):.2f}")

    # aggregation
    def build_updated_metrics(
        self,
        fitness: jnp.ndarray,
        info: Dict[str, Any],
        algo_metrics: Dict[str, Any],
        start_time: float,
        test_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        extra_metrics = {}
        if "accuracy" in info:
            extra_metrics = {
                "best_accuracy_in_generation": info["accuracy"][jnp.argmax(info["accuracy"])],
                "mean_accuracy_in_generation": jnp.mean(info["accuracy"]),
            }

        metrics = {
            **algo_metrics,
            **extra_metrics,
            **test_metrics,
            "gen_time_sec": time.time() - start_time,
            "mean_fitness_in_generation": jnp.mean(fitness),
        }

        # normalize sign when underlying env minimizes
        metrics = {
            k: (-metrics[k] if ('fitness' in k and self.minimize_fitness) else metrics[k])
            for k in MetricsService.METRIC_KEYS
            if k in metrics
        }
        return metrics

    @staticmethod
    def add_cumulative_time(metrics: Dict[str, Any]) -> Dict[str, Any]:
        metrics["cum_gen_time_sec"] = np.cumsum(metrics["gen_time_sec"])
        return metrics
