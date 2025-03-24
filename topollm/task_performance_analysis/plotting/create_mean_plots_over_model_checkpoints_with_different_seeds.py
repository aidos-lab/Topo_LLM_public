import logging
import pathlib
from typing import Any

import numpy as np
from tqdm import tqdm

from topollm.data_processing.dictionary_handling import (
    filter_list_of_dictionaries_by_key_value_pairs,
    generate_fixed_parameters_text_from_dict,
)
from topollm.plotting.line_plot_grouped_by_categorical_column import PlotSizeConfig
from topollm.task_performance_analysis.plotting.distribution_violinplots_and_distribution_boxplots import TicksAndLabels
from topollm.task_performance_analysis.plotting.parameter_combinations_and_loaded_data_handling import (
    add_base_model_data,
    construct_mean_plots_over_model_checkpoints_output_dir_from_filter_key_value_pairs,
    derive_base_model_partial_name,
    get_fixed_parameter_combinations,
)
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def create_mean_plots_over_model_checkpoints_with_different_seeds(
    loaded_data: list[dict],
    array_key_name: str,
    output_root_dir: pathlib.Path,
    plot_size_configs_list: list[PlotSizeConfig],
    *,
    fixed_keys: list[str] | None = None,
    additional_fixed_params: dict[str, Any] | None = None,
    save_plot_raw_data: bool = True,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    if fixed_keys is None:
        fixed_keys = [
            "data_full",
            "data_subsampling_full",
            "data_dataset_seed",
            "model_layer",  # model_layer needs to be an integer
            "model_partial_name",
            "local_estimates_desc_full",
            # Note:
            # - Do NOT fix the model seed, as we want to plot the mean over different seeds.
        ]

    if additional_fixed_params is None:
        additional_fixed_params = {
            "tokenizer_add_prefix_space": "False",  # tokenizer_add_prefix_space needs to be a string
        }

    # Iterate over fixed parameter combinations.
    combinations = list(
        get_fixed_parameter_combinations(
            loaded_data=loaded_data,
            fixed_keys=fixed_keys,
            additional_fixed_params=additional_fixed_params,
        ),
    )
    total_combinations = len(combinations)

    if verbosity >= Verbosity.NORMAL:
        # Log available options
        for fixed_param in fixed_keys:
            options = {entry[fixed_param] for entry in loaded_data if fixed_param in entry}
            logger.info(
                msg=f"{fixed_param=} options: {options=}",  # noqa: G004 - low overhead
            )

    for filter_key_value_pairs in tqdm(
        iterable=combinations,
        total=total_combinations,
        desc="Plotting different choices for model checkpoints",
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

        # The identifier of the base model.
        # This value will be used to select the models for the correlation analysis
        # and add the estimates of the base model for the model checkpoint analysis.
        model_partial_name = filter_key_value_pairs["model_partial_name"]
        base_model_model_partial_name: str = derive_base_model_partial_name(
            model_partial_name=model_partial_name,
        )

        filtered_data_with_added_base_model: list[dict] = add_base_model_data(
            loaded_data=loaded_data,
            base_model_model_partial_name=base_model_model_partial_name,
            filter_key_value_pairs=filter_key_value_pairs,
            filtered_data=filtered_data,
            logger=logger,
        )

        # Sort the arrays by increasing model checkpoint.
        # Then from this point, the list of arrays and list of extracted checkpoints will be in the correct order.
        # 1. Step: Replace None model checkpoints with -1.
        for single_dict in filtered_data_with_added_base_model:
            if single_dict["model_checkpoint"] is None:
                single_dict["model_checkpoint"] = -1
        # 2. Step: Call sorting function.
        sorted_data: list[dict] = sorted(
            filtered_data_with_added_base_model,
            key=lambda single_dict: int(single_dict["model_checkpoint"]),
        )

        extracted_arrays: list[np.ndarray] = [single_dict[array_key_name] for single_dict in sorted_data]
        model_checkpoint_str_list: list[str] = [
            str(object=single_dict["model_checkpoint"]) for single_dict in sorted_data
        ]

        plots_output_dir: pathlib.Path = (
            construct_mean_plots_over_model_checkpoints_output_dir_from_filter_key_value_pairs(
                output_root_dir=output_root_dir,
                filter_key_value_pairs=filter_key_value_pairs,
                verbosity=verbosity,
                logger=logger,
            )
        )

        # Save the sorted data list of dicts with the arrays to a pickle file.
        if save_plot_raw_data:
            pass
            # TODO: Implement the saving

        ticks_and_labels: TicksAndLabels = TicksAndLabels(
            xlabel="checkpoints",
            ylabel=array_key_name,
            xticks_labels=model_checkpoint_str_list,
        )

        for plot_size_config in plot_size_configs_list:
            # # # #
            # Create plots

            # TODO: Implement the actual plotting

            pass
