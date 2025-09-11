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
