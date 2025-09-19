import logging

from topollm.model_inference.perplexity.saving.load_perplexity_containers_from_jsonl_files import (
    load_multiple_perplexity_containers_from_jsonl_files,
)
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import PerplexityContainerSaveFormat, Verbosity
from topollm.typing.types import PerplexityResultsList

default_logger = logging.getLogger(__name__)


def load_perplexity_results(
    embeddings_path_manager: EmbeddingsPathManager,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> PerplexityResultsList:
    """Load the perplexity results from the saved file."""
    save_file_path_josnl = embeddings_path_manager.get_perplexity_container_save_file_absolute_path(
        perplexity_container_save_format=PerplexityContainerSaveFormat.LIST_AS_JSONL,
    )

    loaded_data_list: list[PerplexityResultsList] = load_multiple_perplexity_containers_from_jsonl_files(
        path_list=[
            save_file_path_josnl,
        ],
        verbosity=verbosity,
        logger=logger,
    )

    # Since we are only loading one container, we can directly access the first element
    loaded_data: PerplexityResultsList = loaded_data_list[0]

    return loaded_data
