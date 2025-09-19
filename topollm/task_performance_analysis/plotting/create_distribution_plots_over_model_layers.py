import logging
import pathlib
from typing import Any

import numpy as np
from tqdm import tqdm

from topollm.data_processing.dictionary_handling import (
    dictionary_to_partial_path,
    filter_list_of_dictionaries_by_key_value_pairs,
    generate_fixed_parameters_text_from_dict,
)
from topollm.plotting.plot_size_config import PlotSizeConfigFlat
from topollm.task_performance_analysis.plotting.distribution_violinplots_and_distribution_boxplots import (
    TicksAndLabels,
    make_distribution_boxplots_from_extracted_arrays,
    make_distribution_violinplots_from_extracted_arrays,
)
from topollm.task_performance_analysis.plotting.parameter_combinations_and_loaded_data_handling import (
    get_fixed_parameter_combinations,
)
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def create_distribution_plots_over_model_layers(
    loaded_data: list[dict],
    array_key_name: str,
    output_root_dir: pathlib.Path,
    plot_size_configs_list: list[PlotSizeConfigFlat],
    *,
    fixed_keys: list[str] | None = None,
    additional_fixed_params: dict[str, Any] | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create plots which show the distribution of the local estimates over the layers.

    Here, we want to fix all parameters except for the model_layer.
    This means, we exclude the "model_layer" from the fixed_keys,
    but add the "model_checkpoint" to the fixed_keys.
    """
    if fixed_keys is None:
        fixed_keys = [
            "data_full",
            "data_subsampling_full",
            "model_checkpoint",
            # We want all layers for the given model checkpoint and thus, no filtering for "model_layer" in this case.
            "model_partial_name",
            "model_seed",
            "local_estimates_desc_full",
        ]

    if additional_fixed_params is None:
        additional_fixed_params = {
            "tokenizer_add_prefix_space": "False",  # tokenizer_add_prefix_space needs to be a string
        }

    if verbosity >= Verbosity.NORMAL:
        # Log available options
        for fixed_param in fixed_keys:
            options = {entry[fixed_param] for entry in loaded_data if fixed_param in entry}
            logger.info(
                msg=f"{fixed_param=} options: {options=}",  # noqa: G004 - low overhead
            )

    # Iterate over fixed parameter combinations.
    combinations = list(
        get_fixed_parameter_combinations(
            loaded_data=loaded_data,
            fixed_keys=fixed_keys,
            additional_fixed_params=additional_fixed_params,
        ),
    )
    total_combinations = len(combinations)
    for filter_key_value_pairs in tqdm(
        iterable=combinations,
        total=total_combinations,
        desc="Plotting different choices for the model layers",
    ):
        fixed_params_text: str = generate_fixed_parameters_text_from_dict(
            filters_dict=filter_key_value_pairs,
        )

        filtered_data: list[dict] = filter_list_of_dictionaries_by_key_value_pairs(
            list_of_dicts=loaded_data,
            key_value_pairs=filter_key_value_pairs,
        )

        if len(filtered_data) == 0:
            logger.warning(
                msg=f"No data found for {filter_key_value_pairs = }.",  # noqa: G004 - low overhead
            )
            logger.warning(
                msg="Skipping this combination of parameters.",
            )
            continue

        # Sort the arrays by increasing model layer.
        sorted_data: list[dict] = sorted(
            filtered_data,
            key=lambda single_dict: int(single_dict["model_layer"]),
        )

        extracted_arrays: list[np.ndarray] = [single_dict[array_key_name] for single_dict in sorted_data]
        model_layer_str_list: list[str] = [str(object=single_dict["model_layer"]) for single_dict in sorted_data]

        plots_output_dir: pathlib.Path = pathlib.Path(
            output_root_dir,
            "plots_over_layers",
            dictionary_to_partial_path(
                dictionary=filter_key_value_pairs,
            ),
        )
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{plots_output_dir = }",  # noqa: G004 - low overhead
            )

        ticks_and_labels: TicksAndLabels = TicksAndLabels(
            xlabel="layers",
            ylabel=array_key_name,
            xticks_labels=model_layer_str_list,
        )

        for plot_size_config in plot_size_configs_list:
            # # # #
            # Violin plots
            make_distribution_violinplots_from_extracted_arrays(
                extracted_arrays=extracted_arrays,
                ticks_and_labels=ticks_and_labels,
                fixed_params_text=fixed_params_text,
                plots_output_dir=plots_output_dir,
                plot_size_config=plot_size_config,
                verbosity=verbosity,
                logger=logger,
            )

            # # # #
            # Boxplots
            make_distribution_boxplots_from_extracted_arrays(
                extracted_arrays=extracted_arrays,
                ticks_and_labels=ticks_and_labels,
                fixed_params_text=fixed_params_text,
                plots_output_dir=plots_output_dir,
                plot_size_config=plot_size_config,
                verbosity=verbosity,
                logger=logger,
            )
