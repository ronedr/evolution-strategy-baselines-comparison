import jax
from evosax.problems.rl.brax import BraxProblem


def write_gif_best_running_visualization(algorithm, problem, experiment_path_file, key, state, problem_state) -> None:
    try:
        mean = algorithm._unravel_solution(state.best_solution)
        key, subkey = jax.random.split(key)
        _, problem_state, info = problem.eval(
            subkey, jax.tree.map(lambda x: x[None], mean), problem_state
        )

        if isinstance(problem, BraxProblem):
            rollout = [
                jax.tree_util.tree_map(lambda x: x[0, 0, t], info["env_states"].pipeline_state)
                for t in range(problem.episode_length)
            ]
            from brax.io import html
            html_content = html.render(
                problem.env.sys.tree_replace({"opt.timestep": problem.env.dt}), rollout
            )
            with open(f"{experiment_path_file}.html", "w") as f:
                f.write(html_content)
    except Exception as e:
        print(f"Failed to generate visualization: {e}")
