import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm import tqdm

from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.logging.log_list_info import log_list_info
from topollm.typing.enums import Verbosity
from topollm.typing.types import PerplexityResultsList

if TYPE_CHECKING:
    from topollm.model_inference.perplexity.saving.sentence_perplexity_container import SentencePerplexityContainer

default_logger = logging.getLogger(__name__)


def convert_perplexity_results_list_to_dataframe(
    loaded_data: PerplexityResultsList,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
]:
    """Convert the loaded perplexity results to a pandas dataframe and numpy array."""
    # Empty lists for holding the concatenated data
    token_ids_list: list[int] = []
    token_strings_list: list[str] = []
    token_perplexities_list: list[float] = []

    for _, sentence_perplexity_container in tqdm(
        loaded_data,
        desc="Iterating over loaded_data",
    ):
        sentence_perplexity_container: SentencePerplexityContainer

        token_ids_list.extend(
            sentence_perplexity_container.token_ids,
        )
        token_strings_list.extend(
            sentence_perplexity_container.token_strings,
        )
        token_perplexities_list.extend(
            sentence_perplexity_container.token_perplexities,
        )

    if verbosity >= Verbosity.NORMAL:
        log_list_info(
            token_ids_list,
            list_name="token_ids_list",
            logger=logger,
        )
        log_list_info(
            token_strings_list,
            list_name="token_strings_list",
            logger=logger,
        )
        log_list_info(
            token_perplexities_list,
            list_name="token_perplexities_list",
            logger=logger,
        )

    token_perplexities_df = pd.DataFrame(
        {
            "token_id": token_ids_list,
            "token_string": token_strings_list,
            "token_perplexity": token_perplexities_list,
        },
    )

    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            token_perplexities_df,
            df_name="token_perplexities_df",
            check_for_nan=True,
            logger=logger,
        )

    token_perplexities_array = np.array(
        token_perplexities_list,
    )

    return token_perplexities_df, token_perplexities_array
