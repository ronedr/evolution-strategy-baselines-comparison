# Evolution Strategy Baselines Comparison

A comprehensive benchmarking framework for comparing state-of-the-art Evolution Strategy (ES) algorithms across diverse optimization and reinforcement learning tasks. Built with [JAX](https://github.com/google/jax) and [EvoSax](https://github.com/RobertTLange/evosax) for high-performance, hardware-accelerated evolutionary computation.

## 🎯 Overview

This repository provides a systematic comparison of 7 evolution strategy algorithms across 4 problem domains, enabling reproducible research in black-box optimization and neuroevolution. All experiments leverage JAX's JIT compilation and automatic differentiation for efficient parallel evaluation.

### Supported Algorithms

- **PGPE** - Policy Gradients with Parameter-based Exploration
- **Open_ES** - OpenAI Evolution Strategies
- **SNES** - Separable Natural Evolution Strategies
- **Sep_CMA_ES** - Separable Covariance Matrix Adaptation ES
- **CMA_ES** - Covariance Matrix Adaptation Evolution Strategy
- **DES** - Distributed Evolution Strategies
- **LES** - Learned Evolution Strategies

### Problem Domains

| Domain          | Description                         | Tasks                                      | Metrics                 |
| --------------- | ----------------------------------- | ------------------------------------------ | ----------------------- |
| **BBOB**        | Black-Box Optimization Benchmarking | 24 continuous optimization functions (10D) | Fitness convergence     |
| **Gymnax**      | MinAtar RL Environments             | Asterix, Breakout, Freeway, SpaceInvaders  | Episode return          |
| **Brax**        | Physics-based RL                    | Multiple continuous control tasks          | Episode return          |
| **TorchVision** | Computer Vision                     | MNIST, FashionMNIST, CIFAR10, SVHN         | Classification accuracy |

## 📁 Repository Structure

```
evolution-strategy-baselines-comparison/
├── experiment/                    # Core experiment framework
│   ├── experiment.py             # Main experiment runner
│   ├── run_experiments.py        # Batch experiment orchestration
│   ├── compare_results.py        # Result comparison utilities
│   └── utils/                    # Utilities
│       ├── es_utils.py          # ES algorithm helpers
│       ├── merics_utils.py      # Metrics computation
│       ├── problem_utils.py     # Problem configuration
│       ├── visualisation_utils.py # Visualization tools
│       ├── nn_model.py          # Neural network models
│       └── jax_utils.py         # JAX utilities
├── running_algorithms_main/      # Execution scripts
│   ├── as_scripts/              # Command-line scripts
│   │   ├── bbob_script.py       # BBOB experiments
│   │   ├── gymnax_script.py     # Gymnax experiments
│   │   ├── brax_script.py       # Brax experiments
│   │   ├── vision_script.py     # Vision experiments
│   │   └── jobs/                # SLURM job files
│   └── notebooks/               # Jupyter analysis notebooks
├── results/                      # Experiment outputs (JSON)
├── tuning_results/              # Hyperparameter tuning results
└── ES-tasks-visualization/      # Task visualizations
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/evolution-strategy-baselines-comparison.git
cd evolution-strategy-baselines-comparison

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install jax jaxlib evosax optax
pip install gymnax brax
pip install torch torchvision
pip install plotly tqdm
```

### Running Experiments

#### Option 1: Using Python Scripts Directly

```bash
cd running_algorithms_main/as_scripts

# Run BBOB experiments
python bbob_script.py --es_algorithms "PGPE,Open_ES,SNES"

# Run Gymnax experiments
python gymnax_script.py --es_algorithms "CMA_ES,Sep_CMA_ES"

# Run Brax experiments
python brax_script.py --es_algorithms "DES,LES"

# Run Vision experiments
python vision_script.py --es_algorithms "PGPE,SNES"
```

#### Option 2: Using SLURM (for HPC clusters)

```bash
cd running_algorithms_main/as_scripts

# Submit job
./submit_es.sh <problem_type> <es_algo>

# Examples:
./submit_es.sh bbob PGPE
./submit_es.sh gymnax "Open_ES,SNES"
./submit_es.sh brax CMA_ES
./submit_es.sh vision Sep_CMA_ES

# Grant execution permission if needed
chmod +x submit_es.sh
```

### Analyzing Results

```python
from experiment.compare_results import compare

# Compare algorithms on a specific problem
compare(
    results_dir_path="results",
    folder_path_problem="BBOBProblem/Sphere",
    y_graph="best_fitness",
    x_graph="generation_counter",
    algorithms=["PGPE", "Open_ES", "CMA_ES"]
)
```

Or use the provided Jupyter notebooks:

```bash
cd running_algorithms_main/notebooks
jupyter notebook bbob_notebook.ipynb
```

## 🔧 Configuration

### Experiment Parameters

Each script contains configurable parameters:

```python
# Example from bbob_script.py
num_generations = 1024      # Number of ES generations
population_size = 256       # Population size per generation
eval_batch_size = 256       # Batch size for evaluation
log_period = 100           # Logging frequency
num_dims = 10              # Problem dimensionality
seeds = list(range(0, 5))  # Random seeds for reproducibility
```

### Algorithm-Specific Settings

Algorithms can be configured with custom hyperparameters:

```python
es_dict = {
    "PGPE": {
        "optimizer": optax.adam(learning_rate=0.02),
    },
    "Open_ES": {
        "optimizer": optax.adam(learning_rate=0.05),
        "std_schedule": optax.exponential_decay(
            init_value=0.05,
            transition_steps=1000,
            decay_rate=0.2
        )
    },
    "CMA_ES": {},  # Use default parameters
}
```

## 📊 Metrics Tracked

All experiments track the following metrics:

- `best_fitness` - Best fitness found so far
- `best_fitness_in_generation` - Best fitness in current generation
- `mean_fitness_in_generation` - Mean population fitness
- `mean_fitness_in_generation_test` - Test set performance
- `generation_counter` - Current generation number
- `gen_time_sec` - Time per generation
- `cum_gen_time_sec` - Cumulative time

For vision tasks, additional metrics:

- `best_accuracy_in_generation`
- `mean_accuracy_in_generation`
- `mean_accuracy_in_generation_test`

## 📈 Results Format

Results are saved as JSON files in the following structure:

```
results/
├── BBOBProblem/
│   ├── PGPE/
│   │   ├── 0.json
│   │   ├── 1.json
│   │   └── ...
│   └── CMA_ES/
│       └── ...
├── GymnaxProblem/
│   └── ...
└── ...
```

Each JSON file contains:

```json
{
  "best_fitness": [...],
  "mean_fitness_in_generation": [...],
  "generation_counter": [...],
  "gen_time_sec": [...],
  ...
}
```

## 🧪 Example Use Cases

### 1. Benchmark a New Algorithm

```python
from experiment.experiment import Experiment
from evosax.algorithms import algorithms
import jax

# Define your problem
from evosax.problems import BBOBProblem
problem = BBOBProblem(fn_name="Sphere", num_dims=10)

# Initialize algorithm
key = jax.random.PRNGKey(0)
solution = problem.sample(key)
algorithm = algorithms["PGPE"](
    population_size=256,
    solution=solution
)

# Run experiment
exp = Experiment(
    problem=problem,
    algorithm=algorithm,
    results_dir_path="results",
    seed=0,
    log_period=10
)
metrics = exp.run(num_generations=1000)
```

### 2. Hyperparameter Tuning

See `running_algorithms_main/hyperparamter_tuning.ipynb` for examples.

### 3. Visualize Best Policy

For Brax environments, the framework automatically generates GIF visualizations of the best policy found.

## 🔬 Research Reference

This framework is inspired by and builds upon research in evolution strategies, particularly:

- **"Discovering Evolution Strategies via Meta-Black-Box Optimization"** - Hyperparameter configurations
- **EvoSax** - Core ES implementations
- **JAX** - Hardware acceleration

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add more ES algorithms (e.g., ASEBO, NES variants)
- [ ] Support for additional problem domains
- [ ] Distributed training across multiple GPUs
- [ ] Integration with experiment tracking tools (W&B, MLflow)
- [ ] Statistical significance testing utilities

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@software{evolution_strategy_baselines,
  title={Evolution Strategy Baselines Comparison},
  author={ronedr&eliads},
  year={2025},
  url={https://github.com/ronedr&eliads/evolution-strategy-baselines-comparison}
}
```

## 🙏 Acknowledgments

- [EvoSax](https://github.com/RobertTLange/evosax) - Evolution strategy implementations
- [JAX](https://github.com/google/jax) - High-performance numerical computing
- [Gymnax](https://github.com/RobertTLange/gymnax) - JAX-based RL environments
- [Brax](https://github.com/google/brax) - Physics-based RL environments

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact ronedr@post.bgu.ac.il or eliads@post.bgu.ac.il.

---

**Note**: Before running experiments, update the hardcoded paths in the script files (e.g., `sys.path.append('/home/ronedr/...')`) to match your local setup.
