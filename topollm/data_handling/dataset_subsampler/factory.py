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

"""Factory function to instantiate dataset splitters."""

import logging

from topollm.config_classes.data.data_subsampling_config import DataSubsamplingConfig
from topollm.data_handling.dataset_subsampler import dataset_subsampler_random, dataset_subsampler_take_first
from topollm.data_handling.dataset_subsampler.protocol import DatasetSubsampler
from topollm.typing.enums import DataSamplingMode, Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_dataset_subsampler(
    data_subsampling_config: DataSubsamplingConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> DatasetSubsampler:
    """Return a dataset subsampler."""
    if data_subsampling_config.sampling_mode == DataSamplingMode.TAKE_FIRST:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Using take first dataset subsampling via DatasetSubsamplerTakeFirst.",
            )
        result = dataset_subsampler_take_first.DatasetSubsamplerTakeFirst(
            number_of_samples=data_subsampling_config.number_of_samples,
            verbosity=verbosity,
            logger=logger,
        )
    elif data_subsampling_config.sampling_mode == DataSamplingMode.RANDOM:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg="Using take random dataset subsampling via DatasetSubsamplerRandom.",
            )

        if data_subsampling_config.sampling_seed is None:
            msg: str = (
                f"Unsupported {data_subsampling_config.sampling_seed = } for {data_subsampling_config.sampling_mode = }"
            )
            raise ValueError(
                msg,
            )

        result = dataset_subsampler_random.DatasetSubsamplerRandom(
            number_of_samples=data_subsampling_config.number_of_samples,
            sampling_seed=data_subsampling_config.sampling_seed,
            verbosity=verbosity,
            logger=logger,
        )
    else:
        msg: str = f"Unsupported {data_subsampling_config.sampling_mode = }"
        raise ValueError(
            msg,
        )

    return result
