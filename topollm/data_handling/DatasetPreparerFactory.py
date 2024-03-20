# coding=utf-8
#
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# Standard library imports
import logging

# Third party imports
import datasets

# Local imports
from topollm.config_classes.Configs import DataConfig
from topollm.config_classes.enums import DatasetType
import topollm.data_handling.HuggingfaceDatasetPreparer as HuggingfaceDatasetPreparer
from topollm.data_handling.DatasetPreparerProtocol import DatasetPreparer
from topollm.logging.log_dataset_info import log_huggingface_dataset_info

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_dataset_preparer(
    dataset_type: DatasetType,
    data_config: DataConfig,
    verbosity: int = 1,
    logger: logging.Logger = logging.getLogger(__name__),
) -> DatasetPreparer:
    """
    Factory function to instantiate dataset preparers.
    """
    if dataset_type == DatasetType.HUGGINGFACE_DATASET:
        return HuggingfaceDatasetPreparer.HuggingfaceDatasetPreparer(
            data_config=data_config,
            verbosity=verbosity,
            logger=logger,
        )
    # Extendable to other dataset types
    # elif dataset_type == "unified_format":
    #     return ImageDatasetPreparer(config)
    else:
        raise ValueError(f"Unsupported {dataset_type = }")
