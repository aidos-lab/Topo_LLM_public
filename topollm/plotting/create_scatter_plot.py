# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (mail@ruppik.net)
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
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

"""Create a scatter plots and save to disk."""

import logging
import pathlib

import pandas as pd
import plotly.express as px
from plotly.graph_objs._figure import Figure

from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def create_scatter_plot(
    df: pd.DataFrame,
    output_folder: pathlib.Path | None = None,
    *,
    plot_name: str = "scatter_plot",
    subtitle_text: str | None = None,
    x_column_name: str = "additional_distance_approximate_hausdorff_via_kdtree",
    y_column_name: str = "pointwise_results_np_mean",
    color_column_name: str = "local_estimates_noise_distortion",
    symbol_column_name: str | None = None,
    hover_data: list[str] | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    output_pdf_width: int = 2500,
    ouput_pdf_height: int = 1500,
    show_plot: bool = False,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create an interactive scatter plot using Plotly.

    Args:
        df:
            DataFrame containing the data to plot.
        output_folder:
            Folder for saving the plot.
        show_plot:
            Whether to show the plot.
        verbosity:
            Verbosity level.
        logger:
            Logger instance.

    """
    if hover_data is None:
        hover_data = [
            "experiment_dir_name",
            "local_estimates_noise_seed",
            "local_estimates_noise_distortion",
        ]

    fig: Figure = px.scatter(
        data_frame=df,
        x=x_column_name,
        y=y_column_name,
        color=color_column_name,
        color_continuous_scale="bluered",
        symbol=symbol_column_name,
        hover_data=hover_data,
        title=f"{x_column_name=} vs {y_column_name=}",
        labels={
            x_column_name: x_column_name,
            y_column_name: y_column_name,
            "global_estimate": "global_estimate",
            color_column_name: color_column_name,
        },
    )
    fig.update_traces(
        marker={
            "size": 10,
            "opacity": 0.7,
        },
    )

    if x_min is not None and x_max is not None:
        fig.update_xaxes(
            range=[
                x_min,
                x_max,
            ],
        )
    if y_min is not None and y_max is not None:
        fig.update_yaxes(
            range=[
                y_min,
                y_max,
            ],
        )

    if subtitle_text is not None:
        # Add subtitle
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.05,
            xanchor="center",
            yanchor="bottom",
            text=subtitle_text,
            showarrow=False,
            font={
                "size": 12,
                "color": "black",
            },
        )

    if show_plot:
        fig.show()

    # # # # # # # # # # # # # #
    # Save plots and raw data
    if output_folder is not None:
        output_folder = pathlib.Path(
            output_folder,
        )
        output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save plot as HTML
        output_file_html = pathlib.Path(
            output_folder,
            f"{plot_name}.html",
        )
        output_file_html.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {output_file_html} ...",  # noqa: G004 - low overhead
            )
        fig.write_html(
            output_file_html,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {output_file_html} DONE",  # noqa: G004 - low overhead
            )

        # Save plot as PDF
        output_file_pdf = pathlib.Path(
            output_folder,
            f"{plot_name}.pdf",
        )
        output_file_html.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {output_file_pdf} ...",  # noqa: G004 - low overhead
            )
        fig.write_image(
            file=output_file_pdf,
            format="pdf",
            width=output_pdf_width,
            height=ouput_pdf_height,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving plot to {output_file_pdf} DONE",  # noqa: G004 - low overhead
            )

        # Save the raw data
        output_file_raw_data = pathlib.Path(
            output_folder,
            f"{plot_name}_raw_data.csv",
        )
        output_file_raw_data.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving raw data to {output_file_raw_data} ...",  # noqa: G004 - low overhead
            )
        df.to_csv(
            path_or_buf=output_file_raw_data,
            index=False,
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Saving raw data to {output_file_raw_data} DONE",  # noqa: G004 - low overhead
            )
