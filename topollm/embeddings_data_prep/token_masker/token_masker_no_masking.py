"""Module implementing a TokenMasker that does not perform any masking."""

import logging

import numpy as np

from topollm.config_classes.embeddings_data_prep.token_masking_config import TokenMaskingConfig
from topollm.embeddings_data_prep.prepared_data_containers import PreparedData
from topollm.logging.log_array_info import log_array_info
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class TokenMaskerNoMasking:
    """Implementation of the TokenMasker protocol without masking (which does nothing)."""

    def __init__(
        self,
        token_masking_config: TokenMaskingConfig,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the class."""
        self.token_masking_config: TokenMaskingConfig = token_masking_config

        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

    def mask_tokens(
        self,
        input_data: PreparedData,
    ) -> tuple[
        PreparedData,
        np.ndarray,
    ]:
        """Return the input data without any changes."""
        # This is a no-op masking implementation
        output_data: PreparedData = input_data

        # The indices of the non-masked tokens are all indices
        non_masked_indices: np.ndarray = np.arange(input_data.array.shape[0])

        if self.verbosity >= Verbosity.NORMAL:
            log_array_info(
                array_=non_masked_indices,
                array_name="non_masked_indices",
                logger=self.logger,
            )
            self.logger.info(
                msg=f"{non_masked_indices.shape = }",  # noqa: G004 - low overhead
            )

        return (
            output_data,
            non_masked_indices,
        )
