import brax.envs as brax_envs
import gymnax
from evosax.problems import Problem
from evosax.problems.bbob.bbob import BBOBProblem
from evosax.problems.networks import categorical_output_fn, tanh_output_fn
from evosax.problems.rl.brax import BraxProblem
from evosax.problems.rl.gymnax import GymnaxProblem
from evosax.problems.vision.torchvision import TorchVisionProblem
from gymnax.environments.spaces import Discrete, Box


def get_problem_name(problem: Problem):
    if isinstance(problem, GymnaxProblem) or isinstance(problem, BraxProblem):
        problem_name = problem.env_name
    elif isinstance(problem, BBOBProblem):
        problem_name = problem.fn_name
    elif isinstance(problem, TorchVisionProblem):
        problem_name = problem.task_name
    else:
        raise Exception("No found problem name!!")

    return f"{problem.__class__.__name__}/{problem_name}"


def get_problem_settings(env_name: str):
    """
    Returns an appropriate output function and chosen output dimension
    based on the action space of the given environment name.
    Supports both discrete (gymnax) and continuous (Box) action spaces.
    """
    # Brax environments (via gymnax wrapper)
    if env_name in list(brax_envs._envs.keys()):
        env = brax_envs._envs[env_name]()
        num_actions = env.action_size
        output_fn = tanh_output_fn  # continuous control in Brax
        return num_actions, output_fn
    # Gymnax environments
    elif env_name in gymnax.registered_envs:
        env, env_params = gymnax.make(env_name)
        action_space = env.action_space(env_params)
        if isinstance(action_space, Discrete):
            num_actions = action_space.n
            output_fn = categorical_output_fn
        elif isinstance(action_space, Box):
            num_actions = int(action_space.shape[0])
            output_fn = tanh_output_fn
        else:
            raise ValueError(f"Unsupported action space type: {action_space}")
        return num_actions, output_fn
    else:
        raise ValueError(f"Unrecognized environment name: '{env_name}'")
