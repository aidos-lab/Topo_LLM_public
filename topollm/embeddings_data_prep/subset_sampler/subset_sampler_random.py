"""Implementation of the SubsetSampler protocol for random sampling."""

import logging
from typing import TYPE_CHECKING

import numpy as np

from topollm.config_classes.embeddings_data_prep.sampling_config import SamplingConfig
from topollm.embeddings_data_prep.prepared_data_containers import PreparedData
from topollm.logging.log_array_info import log_array_info
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    import pandas as pd

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class SubsetSamplerRandom:
    """Implementation of the SubsetSampler protocol for random sampling."""

    def __init__(
        self,
        embeddings_data_prep_sampling_config: SamplingConfig,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the SubsetSamplerRandom."""
        self.embeddings_data_prep_sampling_config: SamplingConfig = embeddings_data_prep_sampling_config

        self.rng: np.random.Generator = np.random.default_rng(
            seed=embeddings_data_prep_sampling_config.seed,
        )

        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

    def sample_subsets(
        self,
        input_data: PreparedData,
    ) -> tuple[
        PreparedData,
        np.ndarray,
    ]:
        """Sample a random subset of the rows of the array and the corresponding metadata."""
        array: np.ndarray = input_data.array
        meta_df: pd.DataFrame = input_data.meta_df

        requested_num_samples: int = self.embeddings_data_prep_sampling_config.num_samples

        if len(array) >= requested_num_samples:
            subsample_idx_vector: np.ndarray = self.rng.choice(
                range(len(array)),
                replace=False,
                size=requested_num_samples,
            )
        else:
            self.logger.warning(
                msg=f"{requested_num_samples = } is larger than the number of available samples {array.shape[0] = }.",  # noqa: G004 - low overhead
            )
            subsample_idx_vector: np.ndarray = self.rng.choice(
                range(len(array)),
                replace=False,
                size=len(array),
            )

        subsampled_array: np.ndarray = array[subsample_idx_vector]
        subsampled_df: pd.DataFrame = meta_df.iloc[subsample_idx_vector]

        sampled_data = PreparedData(
            array=subsampled_array,
            meta_df=subsampled_df,
        )

        if self.verbosity >= Verbosity.NORMAL:
            log_array_info(
                array_=subsample_idx_vector,
                array_name="subsample_idx_vector",
                logger=self.logger,
            )
            self.logger.info(
                msg=f"{subsampled_array.shape = }",  # noqa: G004 - low overhead
            )
            self.logger.info(
                msg=f"Expected sample size: {requested_num_samples = }",  # noqa: G004 - low overhead
            )

        return (
            sampled_data,
            subsample_idx_vector,
        )
