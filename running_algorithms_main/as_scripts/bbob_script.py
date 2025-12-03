import sys
sys.path.append('/home/ronedr/evolution-strategy-baselines-comparison')

import optax
from tqdm import tqdm
from evosax.problems import BBOBProblem as Problem, bbob_fns
from experiment.run_experiments import run_experiment_permutations

num_generations = 4000
population_size = 500
eval_batch_size = 256
log_period = 100
num_dims = 10000
seeds = list(range(0, 5))
result_dir = "../../results"
problems_bbob_fns = list(bbob_fns.keys())

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

# all algorithms that we want to comapre with best params according the article "DISCOVERING EVOLUTION STRATEGIES VIA META-BLACK-BOX OPTIMIZATION".
es_dict = {
    "PGPE": {
        "optimizer": optax.adam(learning_rate=0.02),
    },
    "Open_ES": {    
        "optimizer": optax.adam(learning_rate=lr_schedule), 
        "std_schedule": std_schedule
    },
    "SNES": {},
    "Sep_CMA_ES": {},
    "CMA_ES": {},
    "DES": {},
    "LES": {},
}


## Extract the es_algorithms that we want to run in this job.
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--es_algorithms', help="List of es_algorithms that we want to run in this job, example: 'PGPE,Open_ES,ASEBO'", required=True)
args = parser.parse_args()
es_algorithms = args.es_algorithms.split(",")

# take only the es_algorithms we insert as arg.
running_es = {es: es_dict[es] for es in es_algorithms if es in es_dict}

for bbob_fn in tqdm(problems_bbob_fns, desc="Loading Problems .."):
    try:
        problem = Problem(fn_name=bbob_fn, num_dims=num_dims)
        print("Successfully loaded:", bbob_fn)
        run_experiment_permutations(problems=[problem],
                                    es_dict=running_es,
                                    num_generations=num_generations,
                                    population_size=population_size,
                                    result_dir=result_dir, 
                                    run_again_if_exist=False,
                                    log_period=log_period,
                                    eval_batch_size=eval_batch_size,
                                    suffix_experiment_name=f"{population_size}_{num_dims}",
                                    seeds=list(range(0, 5)))
    except Exception as e:
        print("Failed to load:", bbob_fn, e)
        continue
