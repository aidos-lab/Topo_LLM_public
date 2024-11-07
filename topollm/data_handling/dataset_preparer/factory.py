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

"""Factory function to instantiate dataset preparers."""

import logging
from typing import TYPE_CHECKING

from topollm.config_classes.data.data_config import DataConfig
from topollm.data_handling.dataset_preparer import dataset_preparer_huggingface
from topollm.data_handling.dataset_preparer.protocol import DatasetPreparer
from topollm.data_handling.dataset_splitter.factory import get_dataset_splitter
from topollm.typing.enums import DatasetType, Verbosity

if TYPE_CHECKING:
    from topollm.data_handling.dataset_splitter.protocol import DatasetSplitter

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_dataset_preparer(
    data_config: DataConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> DatasetPreparer:
    """Return a dataset preparer for the given dataset type."""
    dataset_splitter: DatasetSplitter = get_dataset_splitter(
        data_splitting_config=data_config.data_splitting,
        verbosity=verbosity,
        logger=logger,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Using {dataset_splitter.__class__.__name__ = } as dataset splitter.",  # noqa: G004 - low overhead
        )

    if data_config.dataset_type in (DatasetType.HUGGINGFACE_DATASET, DatasetType.HUGGINGFACE_DATASET_NAMED_ENTITY):
        result = dataset_preparer_huggingface.DatasetPreparerHuggingface(
            data_config=data_config,
            dataset_splitter=dataset_splitter,
            verbosity=verbosity,
            logger=logger,
        )
    else:
        msg: str = f"Unsupported {data_config.dataset_type = }"
        raise ValueError(msg)

    return result
