# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (mail@ruppik.net)
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

"""Test the compute_and_store_embeddings function."""

import logging

import pytest
import torch

from topollm.compute_embeddings.compute_and_store_embeddings import (
    compute_and_store_embeddings,
)
from topollm.config_classes.main_config import MainConfig


@pytest.mark.uses_transformers_models
@pytest.mark.slow
def test_compute_and_store_embeddings(
    main_config: MainConfig,
    device_fixture: torch.device,
    logger_fixture: logging.Logger,
) -> None:
    """Test the compute_and_store_embeddings function."""
    compute_and_store_embeddings(
        main_config=main_config,
        logger=logger_fixture,
    )
