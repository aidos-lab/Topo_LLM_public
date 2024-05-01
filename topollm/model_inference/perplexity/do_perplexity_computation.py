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

"""Perform the perplexity computation based on the MainConfig object."""

import logging
from typing import TYPE_CHECKING

from topollm.config_classes.main_config import MainConfig
from topollm.data_handling.dataset_preparer.factory import get_dataset_preparer
from topollm.model_handling.prepare_loaded_model_container import prepare_device_and_tokenizer_and_model
from topollm.model_inference.perplexity.compute_perplexity_over_dataset import compute_perplexity_over_dataset

if TYPE_CHECKING:
    import datasets

    from topollm.model_handling.loaded_model_container import LoadedModelContainer

default_logger = logging.getLogger(__name__)


def do_perplexity_computation(
    main_config: MainConfig,
    logger: logging.Logger = default_logger,
) -> None:
    """Run the perplexity computation."""
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Prepare device, tokenizer, model
    loaded_model_container: LoadedModelContainer = prepare_device_and_tokenizer_and_model(
        main_config=main_config,
        logger=logger,
    )
    model = loaded_model_container.model
    # Put model in evaluation mode
    model.eval()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Prepare dataset
    dataset_preparer = get_dataset_preparer(
        data_config=main_config.data,
        verbosity=main_config.verbosity,
        logger=logger,
    )
    dataset: datasets.Dataset = dataset_preparer.prepare_dataset()

    compute_perplexity_over_dataset(
        loaded_model_container=loaded_model_container,
        dataset=dataset,
        column_name=main_config.data.column_name,
        verbosity=main_config.verbosity,
        logger=logger,
    )
