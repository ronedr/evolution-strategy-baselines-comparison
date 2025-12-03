"""
Utilities for visualizing experiment results.

This module provides functions to create convergence plots and other
visualizations from experiment results.
"""

import os
import json
from typing import Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


EPS = 1e-12


def smooth_curve(y, window_size=10, pad_mode="edge"):
    """
    Centered moving average with edge-aware padding.
    
    Parameters
    ----------
    y : array-like
        The curve to smooth
    window_size : int
        Window size for moving average (will be made odd for symmetric centering)
    pad_mode : str
        Padding mode: "edge" (replicate) or "reflect"
        
    Returns
    -------
    np.ndarray
        Smoothed curve
    """
    y = np.asarray(y, dtype=float)
    if window_size < 2 or y.size == 0:
        return y

    if window_size % 2 == 0:
        window_size += 1  # enforce odd window for proper centering

    half = window_size // 2
    y_pad = np.pad(y, (half, half), mode=pad_mode)
    kernel = np.ones(window_size, dtype=float) / window_size
    smoothed = np.convolve(y_pad, kernel, mode="valid")  # same length as y
    return smoothed


def _compute_mean_ci_from_runs(runs, x_key, y_key):
    """
    Align runs by the intersection of x values and compute mean ± 1.96*SE.
    
    Parameters
    ----------
    runs : list of dict
        List of dicts loaded from seed JSONs
    x_key : str
        Key for x-axis values
    y_key : str
        Key for y-axis values
        
    Returns
    -------
    tuple
        (x_aligned, mean, lower, upper)
    """
    series = []
    for r in runs:
        x = r[x_key][:]
        y = r[y_key][:]
        if len(x) != len(y):
            L = min(len(x), len(y))
            x = x[:L]
            y = y[:L]
        series.append(dict(zip(x, y)))

    # common x values across all runs
    common_x = None
    for s in series:
        xs = set(s.keys())
        common_x = xs if common_x is None else (common_x & xs)
    if not common_x:
        raise ValueError("No common x values across runs")

    x_aligned = sorted(common_x)
    Y = np.vstack([[s[xi] for xi in x_aligned] for s in series])
    n = Y.shape[0]

    mean = Y.mean(axis=0)
    std = Y.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
    se = std / np.sqrt(n) if n > 0 else np.zeros_like(mean)
    z = 1.96
    lower = mean - z * se
    upper = mean + z * se
    return np.asarray(x_aligned, dtype=float), mean, lower, upper


def _load_runs_in_dir(dir_path, x_key, y_key):
    """
    Load all JSON seed files from a directory.
    
    Parameters
    ----------
    dir_path : str
        Path to directory containing seed JSON files
    x_key : str
        Key for x-axis values
    y_key : str
        Key for y-axis values
        
    Returns
    -------
    list of dict
        List of loaded run data
    """
    if not os.path.isdir(dir_path):
        return []
    
    files = [f for f in os.listdir(dir_path) if f.endswith(".json")]
    runs = []
    for f in sorted(files):
        with open(os.path.join(dir_path, f), "r") as fh:
            try:
                res = json.load(fh)
                runs.append(res)
            except Exception as e:
                print(f"Error loading {f}: {e}")
            
    runs = [r for r in runs if (x_key in r and y_key in r)]
    return runs


def _mean_curve_from_runs(runs, x_key, y_key):
    """
    Compute mean curve with confidence intervals from multiple runs.
    
    Parameters
    ----------
    runs : list of dict
        List of run data
    x_key : str
        Key for x-axis values
    y_key : str
        Key for y-axis values
        
    Returns
    -------
    tuple or None
        (x, mean, lower, upper) or None if no runs
    """
    if not runs:
        return None
    x, mean, lower, upper = _compute_mean_ci_from_runs(runs, x_key, y_key)
    return x, mean, lower, upper


def _minmax_per_x(curves_dict):
    """
    Normalize curves using min-max normalization per x-value.
    
    This function handles NaN values by computing min/max only from finite values.
    
    Parameters
    ----------
    curves_dict : dict
        {name: (x, y)} for one task at many algos
        
    Returns
    -------
    tuple
        (x_common, {name: y_norm})
    """
    names = list(curves_dict.keys())
    if not names:
        return None, {}

    x_sets = [set(map(float, curves_dict[n][0])) for n in names]
    x_common = sorted(set.intersection(*x_sets))
    if not x_common:
        return None, {}

    y_map = {}
    for n in names:
        x, y = curves_dict[n]
        lut = dict(zip(map(float, x), y))
        y_map[n] = np.array([lut[xi] for xi in x_common], dtype=float)

    Y = np.stack([y_map[n] for n in names], axis=0)  # A x T
    
    # Handle NaN values: compute min/max only from finite values
    # Use nanmin/nanmax to ignore NaN values
    ymin = np.nanmin(Y, axis=0)
    ymax = np.nanmax(Y, axis=0)
    span = np.maximum(ymax - ymin, EPS)
    
    # Normalize each curve, preserving NaN values
    y_norm = {}
    for n in names:
        y_vals = y_map[n]
        # Only normalize finite values
        normalized = (y_vals - ymin) / span
        y_norm[n] = normalized
    
    return np.array(x_common, dtype=float), y_norm


def _auto_colors(names, user_colors):
    """
    Generate color mapping for algorithm names.
    
    Parameters
    ----------
    names : list of str
        Algorithm names
    user_colors : dict or None
        User-provided color mapping
        
    Returns
    -------
    dict
        {name: color}
    """
    if user_colors:
        # Use user colors, fall back to default palette for missing names
        default_palette = px.colors.qualitative.Plotly
        result = {}
        for i, n in enumerate(names):
            if n in user_colors:
                result[n] = user_colors[n]
            else:
                result[n] = default_palette[i % len(default_palette)]
        return result
    
    # Use default palette
    palette = px.colors.qualitative.Plotly
    return {n: palette[i % len(palette)] for i, n in enumerate(names)}


def plot_experiments_convergence_graph(
    results_dir: str,
    algorithm_folders: List[str],
    x_key: str = "generation_counter",
    y_key: str = "best_fitness_in_generation",
    normalize_specific: bool = False,
    show_runs: bool = False,
    smooth_window: int = 10,
    colors: Optional[Dict[str, str]] = None
):
    """
    Plot convergence curves for multiple algorithms across problems.
    
    This function creates interactive Plotly visualizations showing how algorithms
    converge over time. It can handle both single-problem and multi-problem scenarios.
    
    Parameters
    ----------
    results_dir : str
        Path to the root results directory (e.g., "../results")
    algorithm_folders : List[str]
        List of algorithm folder paths relative to results_dir.
        Example: ['BBOBProblem/sphere/CMA_ES_500_1000']
        These paths are typically obtained from filter_algorithm_folders().
    x_key : str
        Key for x-axis values in JSON files (default: "generation_counter")
    y_key : str
        Key for y-axis values in JSON files (default: "best_fitness_in_generation")
    normalize_specific : bool
        If True, apply min-max normalization across algorithms.
        Now handles NaN values properly by computing min/max only from finite values.
    show_runs : bool
        If True, show individual seed runs in addition to mean curves
    smooth_window : int
        Window size for smoothing curves (default: 10)
    colors : dict or None
        Custom color mapping for algorithms, e.g.:
        {"CMA_ES": "blue", "Open_ES": "red"}
        
    Returns
    -------
    None
        Displays the plot using Plotly
        
    Examples
    --------
    # Plot all algorithms for a specific problem
    >>> from experiment.utils import filter_algorithm_folders, plot_experiments_convergence_graph
    >>> folders = filter_algorithm_folders("results", "BBOBProblem", "sphere")
    >>> plot_experiments_convergence_graph(
    ...     results_dir="results",
    ...     algorithm_folders=folders,
    ...     y_key="best_fitness_in_generation"
    ... )
    
    # Plot with custom colors and normalization
    >>> custom_colors = {
    ...     "CMA_ES": "blue",
    ...     "Open_ES": "red",
    ...     "PGPE": "green"
    ... }
    >>> plot_experiments_convergence_graph(
    ...     results_dir="results",
    ...     algorithm_folders=folders,
    ...     normalize_specific=True,
    ...     colors=custom_colors
    ... )
    """
    if not os.path.isdir(results_dir):
        raise ValueError(f"Results directory not found: {results_dir}")
    
    if not algorithm_folders:
        raise ValueError("No algorithm folders provided")
    
    # Parse algorithm folders to group by problem
    # Structure: problem_group/problem_name/algorithm_folder
    problems_map = {}  # problem_name -> {algo_folder: full_path}
    
    for folder_path in algorithm_folders:
        parts = folder_path.replace('\\', '/').split('/')
        
        if len(parts) < 3:
            continue
        
        problem_group = parts[0]
        problem_name = parts[1]
        algorithm_folder = parts[2]
        
        full_path = os.path.join(results_dir, folder_path)
        
        if not os.path.isdir(full_path):
            continue
        
        if problem_name not in problems_map:
            problems_map[problem_name] = {}
        
        problems_map[problem_name][algorithm_folder] = full_path
    
    if not problems_map:
        raise ValueError("No valid algorithm folders found")
    
    # Determine if single problem or multiple problems
    num_problems = len(problems_map)
    
    if num_problems == 1:
        # SINGLE PROBLEM MODE
        problem_name = list(problems_map.keys())[0]
        algo_paths = problems_map[problem_name]
        
        algos = list(algo_paths.keys())
        color_map = _auto_colors(algos, colors)
        curves = {}
        runs_by_algo = {}
        
        for algo, algo_path in algo_paths.items():
            runs = _load_runs_in_dir(algo_path, x_key, y_key)
            if not runs:
                continue
            
            result = _mean_curve_from_runs(runs, x_key, y_key)
            if result is None:
                continue
            
            x, mean, lower, upper = result
            
            if smooth_window and smooth_window > 1:
                mean = smooth_curve(mean, smooth_window)
                lower = smooth_curve(lower, smooth_window)
                upper = smooth_curve(upper, smooth_window)
            
            curves[algo] = (x, mean, lower, upper)
            runs_by_algo[algo] = runs
        
        if not curves:
            raise ValueError("No valid runs found")
        
        if normalize_specific:
            # min-max normalization across algorithms
            x_common, norm_means = _minmax_per_x({k: (v[0], v[1]) for k, v in curves.items()})
            if x_common is None:
                raise ValueError("Could not align x across algorithms for normalization")
        
        fig = go.Figure()
        
        for algo, (x, mean, lower, upper) in curves.items():
            color = color_map.get(algo)
            
            if normalize_specific:
                x = x_common
                mean = norm_means[algo]
                # No CI after normalization
                lower = None
                upper = None
            
            if show_runs:
                for r in runs_by_algo[algo]:
                    xr = r[x_key]
                    yr = r[y_key]
                    L = min(len(xr), len(yr))
                    fig.add_trace(go.Scatter(
                        x=xr[:L],
                        y=yr[:L],
                        mode="lines",
                        line=dict(width=1, dash="dot", color=color),
                        opacity=0.25,
                        name=f"{algo} (run)",
                        legendgroup=algo,
                        showlegend=False,
                        hoverinfo="skip"
                    ))
            
            fig.add_trace(go.Scatter(
                x=x, y=mean, mode="lines", name=algo,
                line=dict(color=color, width=2),
                legendgroup=algo,
                hovertemplate=f"{x_key.replace('_',' ').title()}: "+"%{x}<br>"+
                              ("Norm " if normalize_specific else "") +
                              f"{y_key.replace('_',' ').title()}: "+"%{y:.4f}<extra></extra>"
            ))
            
            if lower is not None and upper is not None:
                fig.add_trace(go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill="toself",
                    fillcolor=color,
                    opacity=0.2,
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=algo
                ))
        
        fig.update_layout(
            title=f"{'Normalized ' if normalize_specific else ''}Comparison on '{problem_name}'",
            xaxis_title=x_key.replace("_", " ").title(),
            yaxis_title=("Normalized " if normalize_specific else "") + y_key.replace("_", " ").title(),
            template="plotly_white",
            hovermode="x unified"
        )
        fig.show()
        
    else:
        # MULTI-PROBLEM MODE: Normalize per problem, then average
        all_algos = set()
        for algo_dict in problems_map.values():
            all_algos.update(algo_dict.keys())
        all_algos = sorted(all_algos)
        
        color_map = _auto_colors(all_algos, colors)
        
        # Per-problem normalization
        per_problem_norm = {}  # problem -> {algo: (x_common, y_norm)}
        
        for problem_name, algo_paths in problems_map.items():
            curves = {}
            
            for algo, algo_path in algo_paths.items():
                runs = _load_runs_in_dir(algo_path, x_key, y_key)
                if not runs:
                    continue
                
                result = _mean_curve_from_runs(runs, x_key, y_key)
                if result is None:
                    continue
                
                x, mean, _, _ = result
                curves[algo] = (x, mean)
            
            if not curves:
                continue
            
            x_common, y_norm = _minmax_per_x(curves)
            if x_common is None:
                continue
            
            per_problem_norm[problem_name] = {a: (x_common, y_norm[a]) for a in y_norm.keys()}
        
        if not per_problem_norm:
            raise ValueError("No valid normalized curves found across problems")
        
        fig = go.Figure()
        
        for algo in all_algos:
            # Gather this algo across all problems where it exists
            series = []
            x_common = None
            
            for problem_name, d in per_problem_norm.items():
                if algo not in d:
                    continue
                
                xs, ys = d[algo]
                xs = np.asarray(xs, dtype=float)
                ys = np.asarray(ys, dtype=float)
                x_common = set(xs) if x_common is None else (x_common & set(xs))
                series.append((xs, ys))
            
            if not series or not x_common:
                continue
            
            x_common = np.array(sorted(x_common), dtype=float)
            
            Y = []
            for xs, ys in series:
                lut = {float(xx): float(yy) for xx, yy in zip(xs, ys)}
                Y.append([lut[xi] for xi in x_common])
            Y = np.array(Y, dtype=float)  # problems x T
            
            mean = np.nanmean(Y, axis=0)  # Use nanmean to handle NaN
            std = np.nanstd(Y, axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros_like(mean)
            se = std / np.sqrt(max(1, Y.shape[0]))
            z = 1.96
            lower = mean - z * se
            upper = mean + z * se
            
            if smooth_window and smooth_window > 1:
                mean = smooth_curve(mean, smooth_window)
                lower = smooth_curve(lower, smooth_window)
                upper = smooth_curve(upper, smooth_window)
            
            color = color_map.get(algo) or px.colors.qualitative.Plotly[0]
            
            fig.add_trace(go.Scatter(
                x=x_common, y=mean, mode="lines", name=algo,
                line=dict(color=color, width=2),
                hovertemplate=f"{x_key.replace('_',' ').title()}: "+"%{x}<br>"+
                              "Avg Norm Perf: "+"%{y:.3f}<extra></extra>"
            ))
            
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_common, x_common[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor=color,
                opacity=0.2,
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False
            ))
        
        fig.update_layout(
            title=f"Average min-max normalized {y_key} over {num_problems} problems",
            xaxis_title=x_key.replace("_", " ").title(),
            yaxis_title="Average normalized performance",
            template="plotly_white",
            hovermode="x unified"
        )
        fig.show()
