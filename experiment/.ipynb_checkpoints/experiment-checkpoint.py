import time
from dataclasses import replace, fields
from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
from evosax.algorithms.base import EvolutionaryAlgorithm
from evosax.problems import Problem, TorchVisionProblem
from evosax.problems.rl.brax import BraxProblem
from evosax.problems.rl.gymnax import GymnaxProblem

from experiment.metrics_service import MetricsService


class Experiment:
    def __init__(
            self,
            problem: Problem,
            algorithm: EvolutionaryAlgorithm,
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
        self.popsize = self._algorithm.population_size
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else self.popsize
        self.log_period = log_period

        self.metrics = MetricsService(
            results_dir_path=results_dir_path,
            problem=problem,
            algorithm=algorithm,
            seed=seed,
            minimize_fitness=self._minimize_fitness,
        )

    def update_params(self, dc, updates: dict):
        names = {f.name for f in fields(dc)}
        safe = {k: v for k, v in updates.items() if k in names}
        if not safe:
            return dc
        return replace(dc, **safe)

    def run(self, num_generations: int) -> Dict[str, Any]:
        key = jax.random.PRNGKey(self._seed)
        key, subkey = jax.random.split(key)

        params = self._algorithm.default_params
        # self._algorithm.elite_ratio = 0.2
        params = self.update_params(params, {"std_init": 0.1})  # 'unknown' ignored

        state = self._algorithm.init(subkey, self._algorithm.solution, params)

        key, subkey = jax.random.split(key)
        problem_state = self._problem.init(subkey)

        collected = []

        for start in range(0, num_generations, self.log_period):
            chunk = min(self.log_period, num_generations - start)
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, chunk)
            (state, params, problem_state), period_gens_metrics = jax.lax.scan(
                self._step,
                (state, params, problem_state),
                keys,
            )
            self.metrics.print_status(chunk + start, period_gens_metrics)
            collected.append(period_gens_metrics)

        metrics = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *collected)
        metrics = self.metrics.add_cumulative_time(metrics)

        self.metrics.save_result(metrics)
        if isinstance(self._problem, BraxProblem):
            self._write_gif_best_running_visualization(key, state, problem_state)

        return metrics

    def has_run(self) -> bool:
        return self.metrics.has_run()

    def get_experiment_path_file(self) -> str:
        return self.metrics.experiment_path_file()

    def _step(self, carry, key):
        state, params, problem_state = carry
        key_ask, key_eval, key_tell = jax.random.split(key, 3)

        start_ts = time.time()

        population, state = self._algorithm.ask(key_ask, state, params)

        fitness, problem_state, info = self._problem.eval(key_eval, population, problem_state)
        # fitness, problem_state, info = self._batched_eval(key_eval, population, problem_state)

        fitness = -fitness if self._minimize_fitness else fitness

        state, algo_metrics = self._algorithm.tell(key_tell, population, fitness, state, params)

        test_metrics = self._eval_test(key_eval, state, problem_state)

        out_metrics = self.metrics.build_updated_metrics(
            fitness=fitness,
            info=info,
            algo_metrics=algo_metrics,
            start_time=start_ts,
            test_metrics=test_metrics,
        )

        return (state, params, problem_state), out_metrics

    def _batched_eval(self, key, population, problem_state) -> Tuple[jnp.ndarray, Any, Dict[str, Any]]:
        num_batches = self.popsize // self.eval_batch_size

        batched_population = jax.tree.map(
            lambda x: x.reshape((num_batches, self.eval_batch_size) + x.shape[1:]),
            population,
        )

        def _eval_batch(carry, batch_idx):
            k, ps = carry
            k, subkey = jax.random.split(k)
            pop_batch = jax.tree.map(lambda x: x[batch_idx], batched_population)
            fitness_batch, new_ps, info_batch = self._problem.eval(subkey, pop_batch, ps)
            fitness_batch = jnp.nan_to_num(fitness_batch, nan=-1e6, posinf=-1e6, neginf=-1e6)
            return (k, new_ps), (fitness_batch, info_batch)

        (_, final_ps), (fitness_all, info_all) = jax.lax.scan(
            _eval_batch, (key, problem_state), jnp.arange(num_batches)
        )

        fitness = fitness_all.reshape((self.popsize,))
        info = jax.tree.map(lambda x: x.reshape((self.popsize,) + x.shape[2:]), info_all)
        return fitness, final_ps, info

    def _eval_test(self, key, state, problem_state) -> Dict[str, Any]:
        mean_solution = self._algorithm.get_mean(state)
        key, key_eval = jax.random.split(key)

        if isinstance(self._problem, TorchVisionProblem):
            fitness, _, info = self._problem.eval_test(
                key_eval,
                jax.tree.map(lambda x: x[None], mean_solution),
                problem_state,
            )
            return {
                "mean_fitness_in_generation_test": fitness.mean(),
                "mean_accuracy_in_generation_test": info["accuracy"].mean(),
            }
        else:
            fitness, _, _ = self._problem.eval(
                key_eval,
                jax.tree.map(lambda x: x[None], mean_solution),
                problem_state,
            )
            fitness = -fitness if self._minimize_fitness else fitness
            return {"mean_fitness_in_generation_test": fitness.mean()}

    # optional visualization
    def _write_gif_best_running_visualization(self, key, state, problem_state) -> None:
        try:
            mean = self._algorithm._unravel_solution(state.best_solution)
            key, subkey = jax.random.split(key)
            _, problem_state, info = self._problem.eval(
                subkey, jax.tree.map(lambda x: x[None], mean), problem_state
            )

            if isinstance(self._problem, BraxProblem):
                rollout = [
                    jax.tree_util.tree_map(lambda x: x[0, 0, t], info["env_states"].pipeline_state)
                    for t in range(self._problem.episode_length)
                ]
                from brax.io import html
                html_content = html.render(
                    self._problem.env.sys.tree_replace({"opt.timestep": self._problem.env.dt}), rollout
                )
                with open(f"{self.metrics.experiment_path_file()}.html", "w") as f:
                    f.write(html_content)
        except Exception as e:
            print(f"Failed to generate visualization: {e}")
