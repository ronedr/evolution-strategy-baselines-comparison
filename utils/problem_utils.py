import gymnax
from evosax.problems import Problem
from evosax.problems.bbob.bbob import BBOBProblem
from evosax.problems.rl.brax import BraxProblem
from evosax.problems.rl.gymnax import GymnaxProblem
from evosax.problems.vision.torchvision import TorchVisionProblem
import brax.envs as brax_envs


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


def get_problem_action_space(env_name: str):
    if env_name in list(brax_envs._envs.keys()):
        return brax_envs._envs[env_name]().action_size
    elif env_name in gymnax.registered_envs:
        env, env_params = gymnax.make(env_name)
        action_space = env.action_space(env_params)
        if hasattr(action_space, "n"):
            return action_space.n
        else:
            return int(action_space.shape[0])
    else:
        raise Exception("No found problem name!!")
