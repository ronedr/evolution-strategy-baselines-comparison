"""
Example usage of the refactored result utilities.

This script demonstrates how to use the new API with filter_algorithm_folders,
summarize_experiments_results_table, and plot_experiments_convergence_graph together.
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiment.utils import (
    filter_algorithm_folders, 
    summarize_experiments_results_table,
    plot_experiments_convergence_graph
)


def example_1_all_problems_in_group():
    """Example 1: Get all algorithms across all problems in TorchVisionProblem."""
    print("=" * 80)
    print("Example 1: All algorithms across all problems in TorchVisionProblem")
    print("=" * 80)
    
    # Filter to get all algorithm folders in TorchVisionProblem
    algorithm_folders = filter_algorithm_folders(
        results_dir="../results",
        problem_group="TorchVisionProblem",
        problem_name=None  # None means all problems
    )
    
    print(f"\nFound {len(algorithm_folders)} algorithm folders:")
    for folder in algorithm_folders[:5]:  # Show first 5
        print(f"  - {folder}")
    if len(algorithm_folders) > 5:
        print(f"  ... and {len(algorithm_folders) - 5} more")
    
    # Create summary table
    table = summarize_experiments_results_table(
        results_dir="../results",
        algorithm_folders=algorithm_folders,
        key="best_accuracy_in_generation",
        func=np.max
    )
    
    print("\nSummary Table:")
    print(table)
    print()


def example_2_specific_algorithm():
    """Example 2: Get only CMA_ES results across all BBOB problems."""
    print("=" * 80)
    print("Example 2: Only CMA_ES across all BBOB problems")
    print("=" * 80)
    
    # Filter for specific algorithm
    algorithm_folders = filter_algorithm_folders(
        results_dir="../results",
        problem_group="BBOBProblem",
        problem_name=None,  # All problems
        algorithms=["CMA_ES"]  # Only CMA_ES
    )
    
    print(f"\nFound {len(algorithm_folders)} CMA_ES folders:")
    for folder in algorithm_folders[:5]:
        print(f"  - {folder}")
    if len(algorithm_folders) > 5:
        print(f"  ... and {len(algorithm_folders) - 5} more")
    
    # Create summary table
    table = summarize_experiments_results_table(
        results_dir="../results",
        algorithm_folders=algorithm_folders,
        key="best_fitness_in_generation",
        func=np.min  # For BBOB, lower is better
    )
    
    print("\nSummary Table:")
    print(table)
    print()


def example_3_specific_parameters():
    """Example 3: Get algorithms with specific parameters."""
    print("=" * 80)
    print("Example 3: Algorithms with population_size=500 across all BBOB problems")
    print("=" * 80)
    
    # Filter for specific parameters
    algorithm_folders = filter_algorithm_folders(
        results_dir="../results",
        problem_group="BBOBProblem",
        problem_name=None,  # All problems
        param_filters=["500", None]  # population_size=500, any num_dims
    )
    
    print(f"\nFound {len(algorithm_folders)} algorithm folders with pop_size=500:")
    for folder in algorithm_folders[:5]:
        print(f"  - {folder}")
    if len(algorithm_folders) > 5:
        print(f"  ... and {len(algorithm_folders) - 5} more")
    
    # Create summary table
    table = summarize_experiments_results_table(
        results_dir="../results",
        algorithm_folders=algorithm_folders,
        key="best_fitness_in_generation",
        func=np.min
    )
    
    print("\nSummary Table:")
    print(table)
    print()


def example_4_single_problem():
    """Example 4: Traditional usage - single problem, all algorithms."""
    print("=" * 80)
    print("Example 4: Single problem (sphere) with all algorithms")
    print("=" * 80)
    
    # Filter for a single problem
    algorithm_folders = filter_algorithm_folders(
        results_dir="../results",
        problem_group="BBOBProblem",
        problem_name="sphere"  # Specific problem
    )
    
    print(f"\nFound {len(algorithm_folders)} algorithm folders for sphere:")
    for folder in algorithm_folders[:5]:
        print(f"  - {folder}")
    if len(algorithm_folders) > 5:
        print(f"  ... and {len(algorithm_folders) - 5} more")
    
    # Create summary table
    table = summarize_experiments_results_table(
        results_dir="../results",
        algorithm_folders=algorithm_folders,
        key="best_fitness_in_generation",
        func=np.min
    )
    
    print("\nSummary Table:")
    print(table)
    print()


def example_5_plot_single_problem():
    """Example 5: Plot convergence curves for a single problem."""
    print("=" * 80)
    print("Example 5: Plot convergence curves for halfcheetah problem")
    print("=" * 80)
    
    # Filter for a single problem
    algorithm_folders = filter_algorithm_folders(
        results_dir="../results",
        problem_group="BraxProblem",
        problem_name="halfcheetah"
    )
    
    print(f"\nFound {len(algorithm_folders)} algorithm folders for halfcheetah")
    print("Generating plot...")
    
    # Custom colors
    custom_colors = {
        "CMA_ES": "blue",
        "DiscoveredES": "red",
        "EvoTF_ES": "green",
        "LearnedES": "orange",
        "Open_ES": "purple",
        "PGPE": "brown",
        "SNES": "pink",
        "SEP_CMA_ES": "cyan",
        "SimpleES": "olive"
    }
    
    # Plot convergence
    plot_experiments_convergence_graph(
        results_dir="../results",
        algorithm_folders=algorithm_folders,
        x_key="generation_counter",
        y_key="best_fitness_in_generation",
        normalize_specific=False,
        show_runs=False,
        colors=custom_colors
    )
    print()


def example_6_plot_with_normalization():
    """Example 6: Plot with normalization across algorithms."""
    print("=" * 80)
    print("Example 6: Plot with normalization for sphere problem")
    print("=" * 80)
    
    # Filter for a single problem
    algorithm_folders = filter_algorithm_folders(
        results_dir="../results",
        problem_group="BBOBProblem",
        problem_name="sphere"
    )
    
    print(f"\nFound {len(algorithm_folders)} algorithm folders for sphere")
    print("Generating normalized plot...")
    
    # Plot with normalization (handles NaN values properly)
    plot_experiments_convergence_graph(
        results_dir="../results",
        algorithm_folders=algorithm_folders,
        x_key="generation_counter",
        y_key="best_fitness_in_generation",
        normalize_specific=True,  # Enable normalization
        show_runs=False
    )
    print()


def example_7_plot_multiple_problems():
    """Example 7: Plot averaged curves across multiple problems."""
    print("=" * 80)
    print("Example 7: Plot averaged curves across all BBOB problems")
    print("=" * 80)
    
    # Filter for all problems in BBOB
    algorithm_folders = filter_algorithm_folders(
        results_dir="../results",
        problem_group="BBOBProblem",
        problem_name=None  # All problems
    )
    
    print(f"\nFound {len(algorithm_folders)} algorithm folders across all BBOB problems")
    print("Generating averaged plot...")
    
    # Plot averaged across problems
    plot_experiments_convergence_graph(
        results_dir="../results",
        algorithm_folders=algorithm_folders,
        x_key="generation_counter",
        y_key="best_fitness_in_generation",
        show_runs=False
    )
    print()


if __name__ == "__main__":
    # Run table examples
    print("\n" + "=" * 80)
    print("TABLE EXAMPLES")
    print("=" * 80 + "\n")
    
    try:
        example_1_all_problems_in_group()
    except Exception as e:
        print(f"Example 1 failed: {e}\n")
    
    try:
        example_2_specific_algorithm()
    except Exception as e:
        print(f"Example 2 failed: {e}\n")
    
    try:
        example_3_specific_parameters()
    except Exception as e:
        print(f"Example 3 failed: {e}\n")
    
    try:
        example_4_single_problem()
    except Exception as e:
        print(f"Example 4 failed: {e}\n")
    
    # Run visualization examples
    print("\n" + "=" * 80)
    print("VISUALIZATION EXAMPLES")
    print("=" * 80 + "\n")
    
    try:
        example_5_plot_single_problem()
    except Exception as e:
        print(f"Example 5 failed: {e}\n")
    
    try:
        example_6_plot_with_normalization()
    except Exception as e:
        print(f"Example 6 failed: {e}\n")
    
    try:
        example_7_plot_multiple_problems()
    except Exception as e:
        print(f"Example 7 failed: {e}\n")

