"""Perform the perplexity computation based on the MainConfig object."""

import logging
from typing import TYPE_CHECKING

from topollm.config_classes.main_config import MainConfig
from topollm.data_handling.dataset_preparer.factory import get_dataset_preparer
from topollm.model_handling.prepare_loaded_model_container import (
    prepare_device_and_tokenizer_and_model_from_main_config,
)
from topollm.model_inference.perplexity.compute_perplexity_over_dataset import (
    compute_perplexity_over_dataset,
)
from topollm.model_inference.perplexity.saving.save_perplexity_results_list import (
    save_perplexity_results_list_in_multiple_formats,
)
from topollm.path_management.embeddings.factory import get_embeddings_path_manager

if TYPE_CHECKING:
    import datasets

    from topollm.model_handling.loaded_model_container import LoadedModelContainer
    from topollm.typing.enums import Verbosity
    from topollm.typing.types import PerplexityResultsList

default_logger = logging.getLogger(__name__)


def do_perplexity_computation(
    main_config: MainConfig,
    logger: logging.Logger = default_logger,
) -> None:
    """Run the perplexity computation."""
    verbosity: Verbosity = main_config.verbosity

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Prepare device, tokenizer, model
    loaded_model_container: LoadedModelContainer = prepare_device_and_tokenizer_and_model_from_main_config(
        main_config=main_config,
        logger=logger,
    )
    model = loaded_model_container.model
    # Put model in evaluation mode
    model.eval()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Prepare dataset
    dataset_preparer = get_dataset_preparer(
        data_config=main_config.data,
        verbosity=verbosity,
        logger=logger,
    )
    dataset: datasets.Dataset = dataset_preparer.prepare_dataset()

    embeddings_path_manager = get_embeddings_path_manager(
        main_config=main_config,
        logger=logger,
    )

    perplexity_results_list: PerplexityResultsList = compute_perplexity_over_dataset(
        loaded_model_container=loaded_model_container,
        dataset=dataset,
        column_name=main_config.data.column_name,
        verbosity=verbosity,
        logger=logger,
    )

    save_perplexity_results_list_in_multiple_formats(
        perplexity_results_list=perplexity_results_list,
        embeddings_path_manager=embeddings_path_manager,
        verbosity=verbosity,
        logger=logger,
    )
