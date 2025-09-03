"""Factory for SubsetSampler."""

import logging

from topollm.config_classes.embeddings_data_prep.sampling_config import SamplingConfig
from topollm.embeddings_data_prep.subset_sampler.protocol import SubsetSampler
from topollm.embeddings_data_prep.subset_sampler.subset_sampler_random import SubsetSamplerRandom
from topollm.embeddings_data_prep.subset_sampler.subset_sampler_take_first import SubsetSamplerTakeFirst
from topollm.typing.enums import EmbeddingsDataPrepSamplingMode, Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_subset_sampler(
    embeddings_data_prep_sampling_config: SamplingConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> SubsetSampler:
    """Get a SubsetSampler instance."""
    match embeddings_data_prep_sampling_config.sampling_mode:
        case EmbeddingsDataPrepSamplingMode.RANDOM:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg="Using random subset sampling via SubsetSamplerRandom.",
                )
            result = SubsetSamplerRandom(
                embeddings_data_prep_sampling_config=embeddings_data_prep_sampling_config,
                verbosity=verbosity,
                logger=logger,
            )
        case EmbeddingsDataPrepSamplingMode.TAKE_FIRST:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg="Using take first subset sampling via SubsetSamplerTakeFirst.",
                )
            result = SubsetSamplerTakeFirst(
                embeddings_data_prep_sampling_config=embeddings_data_prep_sampling_config,
                verbosity=verbosity,
                logger=logger,
            )
        case _:
            msg: str = f"Sampling mode {embeddings_data_prep_sampling_config.sampling_mode} not supported."
            raise ValueError(
                msg,
            )

    return result
