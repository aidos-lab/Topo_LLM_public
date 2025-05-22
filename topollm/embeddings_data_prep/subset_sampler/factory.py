# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (mail@ruppik.net)
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Factory for SubsetSampler."""

import logging

from topollm.config_classes.embeddings_data_prep.sampling_config import EmbeddingsDataPrepSamplingConfig
from topollm.embeddings_data_prep.subset_sampler.protocol import SubsetSampler
from topollm.embeddings_data_prep.subset_sampler.subset_sampler_random import SubsetSamplerRandom
from topollm.embeddings_data_prep.subset_sampler.subset_sampler_take_first import SubsetSamplerTakeFirst
from topollm.typing.enums import EmbeddingsDataPrepSamplingMode, Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_subset_sampler(
    embeddings_data_prep_sampling_config: EmbeddingsDataPrepSamplingConfig,
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
