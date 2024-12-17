"""Analysis of the influence of local estimates for different values of 'n_neighbors'."""

import logging
import pathlib
from typing import Any

import pandas as pd

from topollm.analysis.compare_sampling_methods.make_plots import (
    Y_AXIS_LIMITS,
    analyze_and_plot_influence_of_local_estimates_samples,
)
from topollm.config_classes.constants import NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def retrieve_most_frequent_values(
    full_local_estimates_df: pd.DataFrame,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> dict[str, Any]:
    """Retrieve the most frequent values for fixed parameters.

    This handles cases where mode might be unavailable.
    """
    most_frequent_values = {}
    for column in [
        "data_prep_sampling_method",
        NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["local_estimates_dedup"],
        "data_prep_sampling_seed",
        "data_prep_sampling_samples",
    ]:
        if full_local_estimates_df[column].notna().any():
            most_frequent_values[column] = (
                full_local_estimates_df[column].mode()[0] if not full_local_estimates_df[column].mode().empty else None
            )
        else:
            most_frequent_values[column] = None

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"most_frequent_values:\n{most_frequent_values}",  # noqa: G004 - low overhead
        )

    return most_frequent_values


def analysis_influence_of_local_estimates_n_neighbors(
    full_local_estimates_df: pd.DataFrame,
    array_data_column_name: str,
    results_directory: pathlib.Path,
    selected_subsample_dict: dict | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Run the analysis of local estimates for different values of 'n_neighbors'."""
    if selected_subsample_dict is None:
        selected_subsample_dict = retrieve_most_frequent_values(
            full_local_estimates_df=full_local_estimates_df,
            verbosity=verbosity,
            logger=logger,
        )

    # Run analysis for different values of 'n_neighbors'
    unique_n_neighbors = full_local_estimates_df["n_neighbors"].unique()

    for n_neighbors in unique_n_neighbors:
        for y_min, y_max in Y_AXIS_LIMITS.values():
            common_prefix_path = pathlib.Path(
                results_directory,
                "influence_of_local_estimates_samples",
            )

            plot_save_path: pathlib.Path = pathlib.Path(
                common_prefix_path,
                "plots",
                f"{n_neighbors=}_{y_min=}_{y_max=}.pdf",
            )
            raw_data_save_path: pathlib.Path = pathlib.Path(
                common_prefix_path,
                "raw_data",
                f"{n_neighbors=}.csv",
            )

            analyze_and_plot_influence_of_local_estimates_samples(
                df=full_local_estimates_df,
                n_neighbors=n_neighbors,
                selected_subsample_dict=selected_subsample_dict,
                array_data_column_name=array_data_column_name,
                y_min=y_min,
                y_max=y_max,
                show_plot=False,
                plot_save_path=plot_save_path,
                raw_data_save_path=raw_data_save_path,
                verbosity=verbosity,
                logger=logger,
            )
