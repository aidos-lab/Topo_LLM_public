"""Module for masking tokens of arrays and metadata."""

import logging
from typing import TYPE_CHECKING

import numpy as np

from topollm.config_classes.embeddings_data_prep.token_masking_config import TokenMaskingConfig
from topollm.embeddings_data_prep.prepared_data_containers import PreparedData
from topollm.embeddings_data_prep.token_masker.factory import get_token_masker
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.embeddings_data_prep.token_masker.protocol import TokenMasker

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def mask_tokens_of_arrays_and_meta(
    input_data: PreparedData,
    token_masking_config: TokenMaskingConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> tuple[
    PreparedData,
    np.ndarray,
]:
    """Filter vectors based on a token mask (which is passed in as part of the metadata)."""
    token_masker: TokenMasker = get_token_masker(
        token_masking_config=token_masking_config,
        verbosity=verbosity,
        logger=logger,
    )

    (
        masked_data,
        non_masked_indices,
    ) = token_masker.mask_tokens(
        input_data=input_data,
    )

    # Note: Currently, we do not add information about the non-masked indices to the metadata DataFrame.
    # This can be added in the future if needed.

    return (
        masked_data,
        non_masked_indices,
    )
