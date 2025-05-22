# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (mail@ruppik.net)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go


def plot_statistics_comparison(
    df: pd.DataFrame,
    output_dir: pathlib.Path,
    y_limits: tuple[float, float] | None = None,
    *,
    show_plot: bool = False,
) -> None:
    """Plot token_perplexity and local_estimate over different checkpoints for different models.

    Args:
    ----
        df:
            The DataFrame containing the aggregated statistics.
        output_dir:
            Directory to save the plots.
        y_limits:
            Set the fixed y-axis limits for the plot (min, max).
        show_plot:
            Show the plot in a window.

    """
    # Replace non-finite values in checkpoint with -1 and convert to int
    df["checkpoint"] = (
        pd.to_numeric(
            df["checkpoint"],
            errors="coerce",
        )
        .fillna(-1)
        .astype(int)
    )

    # Sort by checkpoint to ensure smooth lines
    df = df.sort_values(
        by="checkpoint",
    )

    # Unique models to differentiate colors
    unique_models = df["model_without_checkpoint"].unique()

    # Initialize the plot with a larger figure size for better readability in the PDF
    plt.figure(
        figsize=(28, 20),
    )

    # Define markers for different statistics
    markers = {
        "token_perplexity": "o",
        "token_log_perplexity": "x",
        "local_estimate": "s",
    }

    # Use itertools.cycle to iterate over a list of colors.
    # Alternative: plt.cm.viridis.colors for a different palette
    colors = itertools.cycle(plt.cm.tab10.colors)  # type: ignore - colormap is not found

    # Plot data for each model
    for model in unique_models:
        model_df = df[df["model_without_checkpoint"] == model]

        # Use the same color for all three plots of the same model
        color = next(colors)

        plt.plot(
            model_df["checkpoint"],
            model_df["token_perplexity"],
            marker=markers["token_perplexity"],
            linestyle="-",
            label=f"{model} - token_perplexity",
            color=color,
        )
        plt.plot(
            model_df["checkpoint"],
            model_df["token_log_perplexity"],
            marker=markers["token_log_perplexity"],
            linestyle="--",
            label=f"{model} - token_log_perplexity",
            color=color,
        )
        plt.plot(
            model_df["checkpoint"],
            model_df["local_estimate"],
            marker=markers["local_estimate"],
            linestyle="dotted",
            label=f"{model} - local_estimate",
            color=color,
        )

    # Set labels and title
    plt.xlabel("Checkpoint")
    plt.ylabel("Value")
    plt.title("Token Perplexity and Local Estimate Comparison Over Checkpoints")
    plt.legend()

    # Apply fixed scale if provided
    if y_limits:
        plt.ylim(y_limits)

    plt.grid(
        visible=True,
    )

    # Save the plot with increased size as PDF
    output_path_pdf = pathlib.Path(
        output_dir,
        "comparison_plot.pdf",
    )
    plt.savefig(
        output_path_pdf,
        format="pdf",
    )
    if show_plot:
        plt.show()


def plot_statistics_comparison_with_standard_deviation():
    """Plot token_perplexity and local_estimate over different checkpoints for different models with standard deviation shading.

    Inspired by:
    https://stackoverflow.com/questions/61494278/plotly-how-to-make-a-figure-with-multiple-lines-and-shaded-area-for-standard-de
    """
    # sample data in a pandas dataframe
    np.random.seed(1)
    df = pd.DataFrame(
        data={
            "A": np.random.uniform(low=-1, high=2, size=25).tolist(),
            "B": np.random.uniform(low=-4, high=3, size=25).tolist(),
            "C": np.random.uniform(low=-1, high=3, size=25).tolist(),
        },
    )
    df = df.cumsum()

    # define colors as a list
    colors = px.colors.qualitative.Plotly

    # convert plotly hex colors to rgba to enable transparency adjustments
    def hex_rgba(hex, transparency):
        col_hex = hex.lstrip("#")
        col_rgb = list(int(col_hex[i : i + 2], 16) for i in (0, 2, 4))
        col_rgb.extend([transparency])
        areacol = tuple(col_rgb)
        return areacol

    rgba = [hex_rgba(c, transparency=0.2) for c in colors]
    colCycle = ["rgba" + str(elem) for elem in rgba]

    # Make sure the colors run in cycles if there are more lines than colors
    def next_col(cols):
        while True:
            for col in cols:
                yield col

    line_color = next_col(cols=colCycle)

    # plotly  figure
    fig = go.Figure()

    # add line and shaded area for each series and standards deviation
    for i, col in enumerate(df):
        new_col = next(line_color)
        x = list(df.index.values + 1)
        y1 = df[col]
        y1_upper = [(y + np.std(df[col])) for y in df[col]]
        y1_lower = [(y - np.std(df[col])) for y in df[col]]
        y1_lower = y1_lower[::-1]

        # standard deviation area
        fig.add_traces(
            go.Scatter(
                x=x + x[::-1],
                y=y1_upper + y1_lower,
                fill="tozerox",
                fillcolor=new_col,
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                name=col,
            )
        )

        # line trace
        fig.add_traces(
            go.Scatter(
                x=x,
                y=y1,
                line={
                    "color": new_col,
                    "width": 2.5,
                },
                mode="lines",
                name=col,
            ),
        )
    # set x-axis
    fig.update_layout(
        xaxis={
            "range": [1, len(df)],
        },
    )
