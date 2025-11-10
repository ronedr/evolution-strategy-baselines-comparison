import sys
sys.path.append('/home/ronedr/evolution-strategy-baselines-comparison')

import jax
import optax
from tqdm import tqdm
from evosax.problems import CNN, TorchVisionProblem as Problem, identity_output_fn
from experiment.run_experiments import run_experiment_permutations

num_generations = 1000
population_size = 128
log_period = 5
seeds = list(range(0, 5))
result_dir = "../../results"
problems_torch_vision = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]


lr_schedule = optax.exponential_decay(
    init_value=0.01,
    transition_steps=num_generations,
    decay_rate=0.1,
)
std_schedule = optax.exponential_decay(
    init_value=0.05,
    transition_steps=num_generations,
    decay_rate=0.2,
)
es_dict = {
    "Sep_CMA_ES": {},
    "Open_ES": {    
        "optimizer":optax.adam(learning_rate=lr_schedule),
        "std_schedule": std_schedule,
    },
    "SNES": {},
    "PGPE": {
        "optimizer": optax.adam(learning_rate=0.02)
    },
    "LES": {},
    "DES": {},
}

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--es_algorithms', required=True)
args = parser.parse_args()
es_algorithms = args.es_algorithms.split(",")

running_es = {es: es_dict[es] for es in es_algorithms if es in es_dict}

torchvision_batch = 1024

for task_name in tqdm(problems_torch_vision, desc="Loading Problems .."):    
    try:
        problem = Problem(
            task_name=task_name,
            network=CNN(
                num_filters=[16],
                kernel_sizes=[(5, 5)],
                strides=[(1, 1)],
                mlp_layer_sizes=[10],
                output_fn=identity_output_fn
            ),
            batch_size=torchvision_batch
        )
        print("Successfully loaded:", task_name)

        run_experiment_permutations(
            problems=[problem],
            es_dict=running_es,
            num_generations=num_generations,
            population_size=population_size,
            result_dir=result_dir,
            log_period=log_period,
            eval_batch_size=None,
            run_again_if_exist=False,
            seeds=seeds
        )
    except Exception as e:
        print("Failed to run:", task_name, e)
