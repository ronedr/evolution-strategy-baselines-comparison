from evosax.problems import Problem
from evosax.problems.bbob.bbob import BBOBProblem
from evosax.problems.rl.brax import BraxProblem
from evosax.problems.rl.gymnax import GymnaxProblem
from evosax.problems.vision.torchvision import TorchVisionProblem


def get_problem_name(problem: Problem):
    if isinstance(problem, GymnaxProblem) or isinstance(problem, BraxProblem):
        problem_name = problem.env_name
    elif isinstance(problem, BBOBProblem):
        problem_name = problem.fn_name
    elif isinstance(problem, TorchVisionProblem):
        problem_name = problem.task_name
    else:
        raise Exception("No have problem name!!")

    return f"{problem.__class__.__name__}/{problem_name}"
