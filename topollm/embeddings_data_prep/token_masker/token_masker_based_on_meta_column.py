"""Module implementing a TokenMasker that masks tokens based on a metadata column."""

import logging

import numpy as np

from topollm.config_classes.embeddings_data_prep.token_masking_config import TokenMaskingConfig
from topollm.embeddings_data_prep.prepared_data_containers import PreparedData
from topollm.logging.log_array_info import log_array_info
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class TokenMaskerBasedOnMetaColumn:
    """Implementation of the TokenMasker protocol which masks tokens based on a metadata column."""

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
        # Check that the column exists in the metadata DataFrame
        if self.token_masking_config.token_mask_meta_column_name not in input_data.meta_df.columns:
            msg: str = (
                f"Column '{self.token_masking_config.token_mask_meta_column_name=}' not found "
                f"in metadata DataFrame (columns: {input_data.meta_df.columns.tolist()=})."
            )
            self.logger.error(
                msg=msg,
            )
            raise ValueError(msg)

        # TODO: Implement this

        msg = "TokenMaskerBasedOnMetaColumn not implemented yet."
        raise NotImplementedError(
            msg,
        )

        return (
            output_data,
            non_masked_indices,
        )
