# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


"""Container for prepared data."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from topollm.logging.log_array_info import log_array_info
from topollm.logging.log_dataframe_info import log_dataframe_info

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


@dataclass
class PreparedData:
    """Container for prepared data."""

    array: np.ndarray
    meta_df: pd.DataFrame

    def log_info(
        self,
        logger: logging.Logger = default_logger,
    ) -> None:
        log_array_info(
            array_=self.array,
            array_name="array",
            logger=logger,
        )
        log_dataframe_info(
            df=self.meta_df,
            df_name="meta_df",
            logger=logger,
        )
