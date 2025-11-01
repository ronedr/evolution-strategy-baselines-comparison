import json
import os
from typing import Union

from plotly import graph_objects as go


def compare(results_dir_path: str, folder_path_problem: str, y_graph: str, x_graph: str,
            algorithms: Union[list[str], str] = None):
    if algorithms is None or algorithms == "*":
        algorithms = os.listdir(results_dir_path + "/" + folder_path_problem)
    else:
        algorithms = [f"{a}.json" for a in algorithms]

    fig = go.Figure()
    for algorithm in algorithms:
        with open(f"{results_dir_path}/{folder_path_problem}/{algorithm}", "r") as f:
            try:
                metrics_es = json.load(f)
            except:
                continue
            fig.add_trace(go.Scatter(
                x=metrics_es[x_graph],
                y=metrics_es[y_graph],
                mode='lines+markers',
                name=algorithm
            ))
    fig.update_layout(
        title=f'{y_graph} per {x_graph} on {folder_path_problem} function.',
        xaxis_title=f'{x_graph}',
        yaxis_title=f'{y_graph}',
        legend_title='Algorithms',
        template='plotly_white'
    )
    fig.show()
