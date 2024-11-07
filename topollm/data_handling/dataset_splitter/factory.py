# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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

from topollm.config_classes.data.data_splitting_config import DataSplittingConfig
from topollm.data_handling.dataset_splitter import dataset_splitter_do_nothing, dataset_splitter_proportions
from topollm.data_handling.dataset_splitter.protocol import DatasetSplitter
from topollm.typing.enums import DataSplitMode, Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_dataset_splitter(
    data_splitting_config: DataSplittingConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> DatasetSplitter:
    """Return a dataset splitter."""
    if data_splitting_config.data_splitting_mode == DataSplitMode.DO_NOTHING:
        result = dataset_splitter_do_nothing.DatasetSplitterDoNothing(
            verbosity=verbosity,
            logger=logger,
        )
    elif data_splitting_config.data_splitting_mode == DataSplitMode.PROPORTIONS:
        result = dataset_splitter_proportions.DatasetSplitterProportions(
            proportions=data_splitting_config.proportions,
            split_shuffle=data_splitting_config.split_shuffle,
            split_seed=data_splitting_config.split_seed,
            verbosity=verbosity,
            logger=logger,
        )
    else:
        msg: str = f"Unsupported {data_splitting_config.data_splitting_mode = }"
        raise ValueError(msg)

    return result
