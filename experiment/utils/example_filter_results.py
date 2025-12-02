"""
Example script demonstrating the result filtering utilities.

Run this script to see how to filter experiment results by parameters.
"""

from experiment.utils.result_filter_utils import (
    filter_algorithm_folders,
    get_unique_parameter_values,
    parse_algorithm_folder_name
)


def main():
    results_dir = "results"
    problem_group = "BBOBProblem"
    problem_name = "sphere"
    
    print("=" * 70)
    print("Experiment Result Filtering Examples")
    print("=" * 70)
    
    # Example 1: Get all unique population sizes
    print("\n1. Get all unique population sizes (param_index=0):")
    pop_sizes = get_unique_parameter_values(
        results_dir, problem_group, problem_name, param_index=0
    )
    print(f"   Population sizes: {pop_sizes}")
    
    # Example 2: Get all unique num_dims
    print("\n2. Get all unique num_dims (param_index=1):")
    num_dims = get_unique_parameter_values(
        results_dir, problem_group, problem_name, param_index=1
    )
    print(f"   Num dims: {num_dims}")
    
    # Example 3: Filter by specific algorithm
    print("\n3. Get all CMA_ES results:")
    cma_folders = filter_algorithm_folders(
        results_dir, problem_group, problem_name,
        algorithms=["CMA_ES"]
    )
    for folder in cma_folders:
        print(f"   - {folder}")
    
    # Example 4: Filter by population_size=500
    print("\n4. Get all results with population_size=500:")
    pop500_folders = filter_algorithm_folders(
        results_dir, problem_group, problem_name,
        param_filters=["500", None]  # None = any num_dims
    )
    for folder in pop500_folders:
        print(f"   - {folder}")
    
    # Example 5: Filter by specific parameters
    print("\n5. Get all results with population_size=500 and num_dims=1000:")
    specific_folders = filter_algorithm_folders(
        results_dir, problem_group, problem_name,
        param_filters=["500", "1000"]
    )
    for folder in specific_folders:
        print(f"   - {folder}")
    
    # Example 6: Filter by algorithm AND parameters
    print("\n6. Get CMA_ES results with population_size=500:")
    cma_pop500 = filter_algorithm_folders(
        results_dir, problem_group, problem_name,
        algorithms=["CMA_ES"],
        param_filters=["500", None]
    )
    for folder in cma_pop500:
        print(f"   - {folder}")
    
    # Example 7: Parse folder names
    print("\n7. Parse folder name examples:")
    examples = ["CMA_ES_500_1000", "Open_ES_10_10", "PGPE"]
    for example in examples:
        parsed = parse_algorithm_folder_name(example)
        print(f"   '{example}' -> algorithm='{parsed['algorithm']}', params={parsed['params']}")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
