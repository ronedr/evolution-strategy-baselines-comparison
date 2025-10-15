#!/usr/bin/env bash
set -euo pipefail

problem_type=${1:?usage: $0 <problem_type> <es_algo>}
es_algo=${2:?usage: $0 <problem_type> <es_algo>}

# Export into the environment (the comma stays)
export problem_type es_algo

# Now tell Slurm to import those names from the environment
sbatch --export=ALL,problem_type,es_algo \
       --job-name="problem_type-${problem_type}_es_algo-${es_algo}" \
       sbatch_general_experiment.example

