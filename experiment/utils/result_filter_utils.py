"""
Utilities for filtering experiment results by parameters.

This module provides functions to filter experiment result folders based on
algorithm parameters like population_size, num_dims, etc.
"""

import os
from typing import List, Dict, Optional, Any


def parse_algorithm_folder_name(folder_name: str) -> Dict[str, Any]:
    """
    Parse an algorithm folder name to extract algorithm name and parameters.
    
    Expected format: AlgorithmName_param1_param2_...
    Example: "CMA_ES_500_1000" -> {"algorithm": "CMA_ES", "params": ["500", "1000"]}
    
    Args:
        folder_name: Name of the algorithm folder (e.g., "CMA_ES_500_1000")
        
    Returns:
        Dictionary with "algorithm" (str) and "params" (list of str)
    """
    parts = folder_name.split('_')
    
    # Find where the algorithm name ends and parameters begin
    # Parameters are numeric, algorithm names contain letters
    param_start_idx = len(parts)
    for i, part in enumerate(parts):
        if part.isdigit():
            param_start_idx = i
            break
    
    algorithm_name = '_'.join(parts[:param_start_idx])
    params = parts[param_start_idx:]
    
    return {
        "algorithm": algorithm_name,
        "params": params,
        "full_name": folder_name
    }


def filter_algorithm_folders(
    results_dir: str,
    problem_group: str,
    problem_name: str,
    algorithms: Optional[List[str]] = None,
    param_filters: Optional[List[Optional[str]]] = None
) -> List[str]:
    """
    Filter algorithm result folders based on algorithm names and parameter values.
    
    This function searches for experiment result folders matching the specified criteria.
    The folder structure is expected to be:
        results_dir/problem_group/problem_name/algorithm_param1_param2.../seed.json
    
    Args:
        results_dir: Path to results directory (e.g., "results")
        problem_group: Problem group name (e.g., "BBOBProblem", "GymnaxProblem")
        problem_name: Specific problem name (e.g., "sphere", "bent_cigar")
        algorithms: List of algorithm names to filter by. If None, returns all algorithms.
        param_filters: List of parameter values to match positionally.
                      - Each element filters the corresponding parameter position
                      - None means "accept any value" for that position
                      - Example: ["500", None] matches folders with param1=500 and any param2
                      
    Returns:
        List of matching folder paths relative to results_dir
        
    Examples:
        # Get all CMA_ES results for sphere problem
        >>> filter_algorithm_folders("results", "BBOBProblem", "sphere", algorithms=["CMA_ES"])
        ['BBOBProblem/sphere/CMA_ES', 'BBOBProblem/sphere/CMA_ES_500_1000', ...]
        
        # Get all algorithms with population_size=500 and num_dims=1000
        >>> filter_algorithm_folders("results", "BBOBProblem", "sphere", 
        ...                          param_filters=["500", "1000"])
        ['BBOBProblem/sphere/CMA_ES_500_1000', 'BBOBProblem/sphere/Open_ES_500_1000', ...]
        
        # Get all algorithms with population_size=500, any num_dims
        >>> filter_algorithm_folders("results", "BBOBProblem", "sphere",
        ...                          param_filters=["500", None])
        ['BBOBProblem/sphere/CMA_ES_500_1000', 'BBOBProblem/sphere/CMA_ES_500_2000', ...]
    """
    problem_path = os.path.join(results_dir, problem_group, problem_name)
    
    if not os.path.exists(problem_path):
        return []
    
    # Get all algorithm folders
    all_folders = [f for f in os.listdir(problem_path) 
                   if os.path.isdir(os.path.join(problem_path, f))]
    
    matching_folders = []
    
    for folder in all_folders:
        parsed = parse_algorithm_folder_name(folder)
        
        # Filter by algorithm name
        if algorithms is not None:
            if parsed["algorithm"] not in algorithms:
                continue
        
        # Filter by parameters
        if param_filters is not None:
            folder_params = parsed["params"]
            
            # Check if all specified parameter filters match
            matches = True
            for i, filter_value in enumerate(param_filters):
                if filter_value is None:
                    # None means accept any value for this parameter
                    continue
                    
                # Check if folder has this parameter position
                if i >= len(folder_params):
                    matches = False
                    break
                    
                # Check if parameter value matches
                if folder_params[i] != str(filter_value):
                    matches = False
                    break
            
            if not matches:
                continue
        
        # Add relative path
        relative_path = os.path.join(problem_group, problem_name, folder)
        matching_folders.append(relative_path)
    
    return sorted(matching_folders)


def get_unique_parameter_values(
    results_dir: str,
    problem_group: str,
    problem_name: str,
    param_index: int,
    algorithms: Optional[List[str]] = None
) -> List[str]:
    """
    Get all unique values for a specific parameter position across algorithm folders.
    
    Useful for discovering what parameter values exist in your results.
    
    Args:
        results_dir: Path to results directory
        problem_group: Problem group name
        problem_name: Specific problem name
        param_index: Which parameter position to extract (0-indexed)
        algorithms: Optional list of algorithms to filter by
        
    Returns:
        Sorted list of unique parameter values at the specified position
        
    Example:
        # Get all unique population sizes (assuming it's the first parameter)
        >>> get_unique_parameter_values("results", "BBOBProblem", "sphere", param_index=0)
        ['10', '500']
        
        # Get all unique num_dims (assuming it's the second parameter)
        >>> get_unique_parameter_values("results", "BBOBProblem", "sphere", param_index=1)
        ['10', '1000', '2000', '5000', '10000']
    """
    problem_path = os.path.join(results_dir, problem_group, problem_name)
    
    if not os.path.exists(problem_path):
        return []
    
    all_folders = [f for f in os.listdir(problem_path) 
                   if os.path.isdir(os.path.join(problem_path, f))]
    
    unique_values = set()
    
    for folder in all_folders:
        parsed = parse_algorithm_folder_name(folder)
        
        # Filter by algorithm if specified
        if algorithms is not None and parsed["algorithm"] not in algorithms:
            continue
        
        # Extract parameter value at specified index
        if param_index < len(parsed["params"]):
            unique_values.add(parsed["params"][param_index])
    
    return sorted(list(unique_values))


def filter_by_named_params(
    results_dir: str,
    problem_group: str,
    problem_name: str,
    algorithms: Optional[List[str]] = None,
    **param_kwargs
) -> List[str]:
    """
    Filter algorithm folders using named parameters for better readability.
    
    This is a convenience wrapper around filter_algorithm_folders that uses
    keyword arguments instead of positional parameter lists.
    
    Args:
        results_dir: Path to results directory
        problem_group: Problem group name
        problem_name: Specific problem name
        algorithms: Optional list of algorithm names
        **param_kwargs: Named parameters (e.g., population_size=500, num_dims=1000)
                       Parameter names should match the order in suffix_experiment_name
                       
    Returns:
        List of matching folder paths
        
    Example:
        # For BBOB where suffix is "{population_size}_{num_dims}"
        >>> filter_by_named_params("results", "BBOBProblem", "sphere",
        ...                        population_size=500, num_dims=1000)
        
        # For Vision where suffix might be "{population_size}_{dataset}"
        >>> filter_by_named_params("results", "TorchVisionProblem", "MNIST",
        ...                        population_size=256, dataset="MNIST")
    """
    # Note: This function requires knowing the parameter order
    # For now, we'll just pass through the values in the order they were provided
    # A more robust implementation would require a configuration mapping
    
    param_values = list(param_kwargs.values())
    return filter_algorithm_folders(
        results_dir, problem_group, problem_name, 
        algorithms=algorithms,
        param_filters=param_values
    )
