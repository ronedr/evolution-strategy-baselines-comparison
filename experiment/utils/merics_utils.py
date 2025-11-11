import time
from typing import Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
from evosax.algorithms.distribution_based.base import State, Params
from evosax.problems import Problem, TorchVisionProblem
from evosax.types import Fitness, Metrics, Population

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


def print_status(problem: Problem, generation_count: int, metrics: Dict[str, Any]) -> None:
    print(
        f"Generation {generation_count:03d}"
        f" | Best fitness (Train): {float(metrics['best_fitness'][-1]):.2f}"
        f" | Best fitness in generation (Train): {float(metrics['best_fitness_in_generation'][-1]):.2f}"
        f" | Mean fitness (Train): {float(metrics['mean_fitness_in_generation'][-1]):.2f}"
        f" | Mean fitness (Test): {float(metrics['mean_fitness_in_generation_test'][-1]):.2f}"
    )
    if isinstance(problem, TorchVisionProblem):
        print(f" | Mean Accuracy (Test): {float(metrics['mean_accuracy_in_generation_test'][-1]):.2f}")


def custom_metrics_fn(key: jax.Array,
                      population: Population,
                      fitness: Fitness,
                      state: State,
                      params: Params,
                      ) -> Metrics:
    best_idx_in_generation = jnp.argmin(fitness)
    return {
        "generation_counter": state.generation_counter,
        "best_fitness_in_generation": fitness[best_idx_in_generation],
        "mean_fitness_in_generation": fitness.mean(),
        "best_solution_in_generation": population[best_idx_in_generation],
        "best_fitness": state.best_fitness,
        "best_solution": state.best_solution,
        "best_solution_norm": jnp.linalg.norm(state.best_solution),
    }


# aggregation
def build_updated_metrics(fitness: jnp.ndarray, info: Dict[str, Any], algo_metrics: Dict[str, Any], start_time: float,
                          test_metrics: Dict[str, Any], minimize_fitness: bool, metric_keys=None) -> Dict[str, Any]:
    if metric_keys is None:
        metric_keys = METRIC_KEYS

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
        k: (-metrics[k] if ('fitness' in k and minimize_fitness) else metrics[k])
        for k in metric_keys
        if k in metrics
    }
    return metrics


def add_cumulative_time(metrics: Dict[str, Any]) -> Dict[str, Any]:
    metrics["cum_gen_time_sec"] = np.cumsum(metrics["gen_time_sec"])
    return metrics
