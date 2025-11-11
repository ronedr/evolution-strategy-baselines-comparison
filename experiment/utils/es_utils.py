import time
from functools import partial
from typing import Dict, Any, Tuple

import jax
from evosax.algorithms.distribution_based.base import DistributionBasedAlgorithm
from evosax.problems import Problem
from evosax.problems.vision.torchvision import TorchVisionProblem
from jax import numpy as jnp

from experiment.utils.merics_utils import build_updated_metrics


def eval_test(algorithm, problem, minimize_fitness, key, state, problem_state) -> Dict[str, Any]:
    mean_solution = algorithm.get_mean(state)
    key, key_eval = jax.random.split(key)

    if isinstance(problem, TorchVisionProblem):
        fitness, _, info = problem.eval_test(
            key_eval,
            jax.tree.map(lambda x: x[None], mean_solution),
            problem_state,
        )
        return {
            "mean_fitness_in_generation_test": fitness.mean(),
            "mean_accuracy_in_generation_test": info["accuracy"].mean(),
        }
    else:
        fitness, _, _ = problem.eval(
            key_eval,
            jax.tree.map(lambda x: x[None], mean_solution),
            problem_state,
        )
        fitness = -fitness if minimize_fitness else fitness
        return {"mean_fitness_in_generation_test": fitness.mean()}


def batched_eval(algorithm: DistributionBasedAlgorithm, problem, eval_batch_size, key, population,
                 problem_state) -> Tuple[jnp.ndarray, Any, Dict[str, Any]]:
    num_batches = algorithm.population_size // eval_batch_size

    batched_population = jax.tree.map(
        lambda x: x.reshape((num_batches, eval_batch_size) + x.shape[1:]),
        population,
    )

    def _eval_batch(carry, batch_idx):
        k, ps = carry
        k, subkey = jax.random.split(k)
        pop_batch = jax.tree.map(lambda x: x[batch_idx], batched_population)
        fitness_batch, new_ps, info_batch = problem.eval(subkey, pop_batch, ps)
        fitness_batch = jnp.nan_to_num(fitness_batch, nan=-1e6, posinf=-1e6, neginf=-1e6)
        return (k, new_ps), (fitness_batch, info_batch)

    (_, final_ps), (fitness_all, info_all) = jax.lax.scan(
        _eval_batch, (key, problem_state), jnp.arange(num_batches)
    )

    fitness = fitness_all.reshape((algorithm.population_size,))
    info = jax.tree.map(lambda x: x.reshape((algorithm.population_size,) + x.shape[2:]), info_all)
    return fitness, final_ps, info


def step(algorithm: DistributionBasedAlgorithm,
         problem: Problem,
         minimize_fitness: bool,
         carry,
         key):
    state, params, problem_state = carry
    k1, k2, k3 = jax.random.split(key, 3)

    start_ts = time.time()
    population, state = algorithm.ask(k1, state, params)
    fitness, problem_state, info = problem.eval(k2, population, problem_state)
    # fitness, problem_state, info = batched_eval(algorithm, problem, 128, k2, population, problem_state)
    fitness = -fitness if minimize_fitness else fitness
    state, metrics = algorithm.tell(k3, population, fitness, state, params)

    test_metrics = eval_test(algorithm, problem, minimize_fitness, key, state, problem_state)

    out_metrics = build_updated_metrics(
        fitness=fitness,
        info=info,
        algo_metrics=metrics,
        start_time=start_ts,
        test_metrics=test_metrics,
        minimize_fitness=minimize_fitness
    )

    return (state, params, problem_state), out_metrics


@partial(
    jax.jit,
    static_argnames=("algorithm", "problem", "minimize_fitness"),
    donate_argnums=(3,),
)
def scan_step(algorithm, problem, minimize_fitness, carry, keys):
    def scan_step_fn(c, k):
        # c is the carry, do not assign to outer `carry` here!
        return step(algorithm, problem, minimize_fitness, c, k)

    return jax.lax.scan(scan_step_fn, carry, keys)
