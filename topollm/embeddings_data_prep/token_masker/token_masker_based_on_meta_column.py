"""Module implementing a TokenMasker that masks tokens based on a metadata column."""

import logging

import numpy as np
import pandas as pd

from topollm.config_classes.embeddings_data_prep.token_masking_config import TokenMaskingConfig
from topollm.embeddings_data_prep.prepared_data_containers import PreparedData
from topollm.logging.log_array_info import log_array_info
from topollm.logging.log_dataframe_info import log_dataframe_info
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
        """Mask tokens based on a metadata column."""
        array: np.ndarray = input_data.array
        meta_df: pd.DataFrame = input_data.meta_df

        # Check that the column exists in the metadata DataFrame
        if self.token_masking_config.token_mask_meta_column_name not in meta_df.columns:
            msg: str = (
                f"Column '{self.token_masking_config.token_mask_meta_column_name=}' not found "
                f"in metadata DataFrame (columns: {meta_df.columns.tolist()=})."
            )
            self.logger.error(
                msg=msg,
            )
            raise ValueError(msg)

        # Only keep array rows and meta rows where the column is non-zero / True
        row_indices_to_keep: pd.Series = meta_df[self.token_masking_config.token_mask_meta_column_name].astype(bool)
        masked_array: np.ndarray = array[row_indices_to_keep.to_numpy(), :]
        masked_meta_df: pd.DataFrame = meta_df[row_indices_to_keep]

        non_masked_indices: np.ndarray = np.where(row_indices_to_keep)[0]

        masked_data = PreparedData(
            array=masked_array,
            meta_df=masked_meta_df,
        )

        if self.verbosity >= Verbosity.NORMAL:
            log_array_info(
                array_=masked_array,
                array_name="masked_array",
                logger=self.logger,
            )
            log_dataframe_info(
                df=masked_meta_df,
                df_name="masked_meta_df",
                logger=self.logger,
            )
            self.logger.info(
                msg=f"Before masking:\n\t{array.shape = }\n\t{meta_df.shape = }",  # noqa: G004 - low overhead
            )

            self.logger.info(
                msg=f"After masking:\n\t{masked_array.shape = }\n\t{masked_meta_df.shape = }",  # noqa: G004 - low overhead
            )

        return (
            masked_data,
            non_masked_indices,
        )
