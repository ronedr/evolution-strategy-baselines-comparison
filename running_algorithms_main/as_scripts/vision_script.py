# === Runtime safety flags MUST be set before importing jax ===
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".85"
# allow fallback conv algos and lower autotune aggressiveness
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                           " --xla_gpu_strict_conv_algorithm_picker=false"
                           " --xla_gpu_autotune_level=1")
# optional, can help avoid flaky autotune paths
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# === add to sys the main folder. ===
import sys
sys.path.append('/home/ronedr/evolution-strategy-baselines-comparison')

# === Extract the es_algorithms that we want to run in this job. ===
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--es_algorithms', help="List of es_algorithms that we want to run in this job, example: 'PGPE,Open_ES,ASEBO'", required=True)
args = parser.parse_args()
es_algorithms = args.es_algorithms.split(",")

# === imports. ===
import gc
import jax
import optax
from tqdm import tqdm
from evosax.problems import CNN, TorchVisionProblem as Problem, identity_output_fn
from experiment.run_experiments import run_experiment_permutations
from evosax.core.fitness_shaping import standardize_fitness_shaping_fn

num_generations = 1000
population_size = 128
seeds = list(range(0, 5))
result_dir = "../../results"
problems_torch_vision = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]

# all algorithms that we want to compare with best params according to the article "DISCOVERING EVOLUTION STRATEGIES VIA META-BLACK-BOX OPTIMIZATION".
es_dict = {
    "PGPE": {
        "optimizer": optax.adam(learning_rate=0.02)
    },
    "ASEBO": {
        "optimizer": optax.adam(learning_rate=0.01)
    },
    "LES": {},
    "Open_ES": {
        "optimizer": optax.adam(learning_rate=0.05)
    },
    "SNES": {},
    "Sep_CMA_ES": {},
    "CMA_ES": {},
    "DES": {},
}

# take only the es_algorithms we insert as arg.
running_es = {es: es_dict[es] for es in es_algorithms if es in es_dict}

# Use a modest per-step dataloader batch so conv shapes stay realistic during autotune.
# You can bump this after things are stable.
torchvision_batch = 128

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
            run_again_if_exist=False,
            seeds=seeds
        )
    except Exception as e:
        print("Failed to load:", task_name, e)
        continue
    finally:
        # aggressively drop references and compiled executables between tasks
        try:
            del problem
        except NameError:
            pass
        jax.clear_caches()
        gc.collect()
