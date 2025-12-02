"""
Utilities for summarizing experiment results into tables.

This module provides functions to create summary tables from experiment results,
aggregating metrics across seeds and organizing by problems and algorithms.
"""

import os
import json
from typing import Callable, Dict, List, Tuple, Any
import numpy as np
import pandas as pd


def summarize_experiments_results_table(
    results_dir: str,
    algorithm_folders: List[str],
    key: str,
    func: Callable[[List[float]], float],
    fmt: str = "{:.4g}±{:.2g}",
    ddof: int = 1
) -> pd.DataFrame:
    """
    Build a Problems × Algorithms table of mean±std across seeds.
    
    This function takes a list of algorithm folder paths (as returned by 
    filter_algorithm_folders) and creates a summary table showing the 
    aggregated results across different problems and algorithms.
    
    Parameters
    ----------
    results_dir : str
        Path to the root results directory (e.g., "../results").
        Algorithm folder paths are relative to this directory.
    algorithm_folders : List[str]
        List of algorithm folder paths relative to results_dir.
        Example: ['BBOBProblem/sphere/CMA_ES_500_1000', 
                  'BBOBProblem/sphere/Open_ES_500_1000']
        These paths are typically obtained from filter_algorithm_folders().
    key : str
        The JSON key whose value is a list of numbers (per generation).
        Examples: "best_fitness", "best_accuracy_in_generation", "gen_time_sec"
    func : Callable
        A reducer applied to the list from each seed JSON (e.g., np.max, np.min, np.mean).
        For each seed file: value_for_seed = func(data[key]).
    fmt : str
        Format string for cell rendering as mean±std (default: "{:.4g}±{:.2g}").
    ddof : int
        Delta degrees of freedom for std (default 1 gives sample std).
    
    Returns
    -------
    pd.DataFrame
        Rows are problems, columns are algorithms, cells are formatted mean±std strings.
        
    Examples
    --------
    # Get all CMA_ES results and create a summary table
    >>> from experiment.utils.result_filter_utils import filter_algorithm_folders
    >>> folders = filter_algorithm_folders("results", "BBOBProblem", 
    ...                                     problem_name=None, algorithms=["CMA_ES"])
    >>> table = summarize_experiments_results_table(
    ...     results_dir="results",
    ...     algorithm_folders=folders,
    ...     key="best_fitness_in_generation",
    ...     func=np.max
    ... )
    
    # Get all algorithms for specific problems with specific parameters
    >>> folders = filter_algorithm_folders("results", "TorchVisionProblem",
    ...                                     problem_name=None)
    >>> table = summarize_experiments_results_table(
    ...     results_dir="results",
    ...     algorithm_folders=folders,
    ...     key="best_accuracy_in_generation",
    ...     func=np.max
    ... )
    """
    if not os.path.isdir(results_dir):
        raise ValueError(f"Results directory not found: {results_dir}")
    
    # Collect per (problem, algo) -> list of per-seed reduced values
    results: Dict[Tuple[str, str], List[float]] = {}
    
    for folder_path in algorithm_folders:
        # Parse the folder path: problem_group/problem_name/algorithm_folder
        parts = folder_path.replace('\\', '/').split('/')
        
        if len(parts) < 3:
            # Skip invalid paths
            continue
        
        problem_group = parts[0]
        problem_name = parts[1]
        algorithm_folder = parts[2]
        
        # Full path to the algorithm folder
        algo_path = os.path.join(results_dir, folder_path)
        
        if not os.path.isdir(algo_path):
            continue
        
        # Process all seed JSON files in this algorithm folder
        per_seed_vals: List[float] = []
        
        for fname in sorted(os.listdir(algo_path)):
            if not fname.lower().endswith(".json"):
                continue
            
            fpath = os.path.join(algo_path, fname)
            
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                # Skip unreadable or malformed files
                continue
            
            if key not in data:
                # Skip if key is missing
                continue
            
            seq = data[key]
            if not isinstance(seq, list) or len(seq) == 0:
                continue
            
            try:
                value = float(func(seq))
            except Exception:
                # If func fails, skip this seed
                continue
            
            if np.isfinite(value):
                per_seed_vals.append(value)
        
        if per_seed_vals:
            # Store results with problem_name as the problem identifier
            # and algorithm_folder as the algorithm identifier
            results[(problem_name, algorithm_folder)] = per_seed_vals
    
    if not results:
        # Return empty DataFrame if no results found
        return pd.DataFrame()
    
    # Build a Problems × Algorithms DataFrame of formatted mean±std
    problems = sorted({p for (p, _) in results.keys()})
    algos = sorted({a for (_, a) in results.keys()})
    
    table = pd.DataFrame(index=problems, columns=algos, dtype=object)
    
    for p in problems:
        for a in algos:
            vals = results.get((p, a))
            if not vals:
                table.loc[p, a] = ""
                continue
            
            vals_arr = np.asarray(vals, dtype=float)
            mean = float(np.mean(vals_arr))
            std = float(np.std(vals_arr, ddof=ddof)) if len(vals_arr) > 1 else 0.0
            table.loc[p, a] = fmt.format(mean, std)
    
    table.index.name = "Problem"
    table.columns.name = "Algorithm"
    return table
