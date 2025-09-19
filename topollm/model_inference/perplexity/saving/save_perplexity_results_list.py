"""Save the perplexity results list to a file."""

import logging
import pathlib
import pickle

import numpy as np
import pandas as pd
import zarr

from topollm.model_inference.perplexity.saving.sentence_perplexity_container import SentencePerplexityContainer
from topollm.path_management.embeddings.protocol import EmbeddingsPathManager
from topollm.typing.enums import PerplexityContainerSaveFormat, Verbosity
from topollm.typing.types import PerplexityResultsList

default_logger = logging.getLogger(__name__)


def save_perplexity_results_list_as_pickle(
    perplexity_results_list: PerplexityResultsList,
    save_file_path: pathlib.Path,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save the perplexity results list as pickle."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"Saving perplexity results to {save_file_path = } ...",  # noqa: G004 - low overhead
        )
    with pathlib.Path(
        save_file_path,
    ).open(
        mode="wb",
    ) as file:
        pickle.dump(
            obj=perplexity_results_list,
            file=file,
        )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"Saving perplexity results to {save_file_path = } DONE",  # noqa: G004 - low overhead
        )


def save_perplexity_results_list_as_jsonl(
    perplexity_results_list: PerplexityResultsList,
    save_file_path: pathlib.Path,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save the perplexity results list as jsonl."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"Saving perplexity results to {save_file_path = } ...",  # noqa: G004 - low overhead
        )
    with pathlib.Path(
        save_file_path,
    ).open(
        mode="w",
    ) as file:
        # Iterate over the list and save each item as a jsonl line
        for _, sentence_perplexity_container in perplexity_results_list:
            if not isinstance(
                sentence_perplexity_container,
                SentencePerplexityContainer,
            ):
                msg = "Expected a SentencePerplexityContainer."
                raise TypeError(msg)

            model_dump: str = sentence_perplexity_container.model_dump_json()
            file.write(model_dump)
            file.write("\n")
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"Saving perplexity results to {save_file_path = } DONE",  # noqa: G004 - low overhead
        )


def save_perplexity_array_as_zarr(
    perplexities_array: np.ndarray,
    save_file_path: pathlib.Path,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save the perplexities array to a zarr file."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{save_file_path = }",  # noqa: G004 - low overhead
        )
        logger.info(
            "Saving perplexities_array to zarr file ...",
        )
    zarr.save(
        str(save_file_path),
        perplexities_array,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "Saving perplexities_array to zarr file DONE",
        )


def save_perplexity_df_as_csv(
    perplexities_df: pd.DataFrame,
    save_file_path: pathlib.Path,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save the perplexity dataframe to a csv file."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{save_file_path = }",  # noqa: G004 - low overhead
        )
        logger.info(
            "Saving dataframe to csv file ...",
        )
    perplexities_df.to_csv(
        save_file_path,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "Saving dataframe to csv file DONE",
        )


def save_perplexity_results_list_in_multiple_formats(
    perplexity_results_list: PerplexityResultsList,
    embeddings_path_manager: EmbeddingsPathManager,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Save the perplexity results list to a file."""
    for perplexity_container_save_format in [
        PerplexityContainerSaveFormat.LIST_AS_PICKLE,
        PerplexityContainerSaveFormat.LIST_AS_JSONL,
    ]:
        save_file_path = embeddings_path_manager.get_perplexity_container_save_file_absolute_path(
            perplexity_container_save_format=perplexity_container_save_format,
        )

        save_file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        match perplexity_container_save_format:
            case PerplexityContainerSaveFormat.LIST_AS_PICKLE:
                save_perplexity_results_list_as_pickle(
                    perplexity_results_list=perplexity_results_list,
                    save_file_path=save_file_path,
                    verbosity=verbosity,
                    logger=logger,
                )
            case PerplexityContainerSaveFormat.LIST_AS_JSONL:
                save_perplexity_results_list_as_jsonl(
                    perplexity_results_list=perplexity_results_list,
                    save_file_path=save_file_path,
                    verbosity=verbosity,
                    logger=logger,
                )
            case _:
                msg = "Unsupported perplexity container save format."
                raise ValueError(msg)
