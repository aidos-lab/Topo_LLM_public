# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
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

import logging
import pprint

import omegaconf
from hydra import compose, initialize_config_module

from topollm.config_classes.embeddings.embeddings_config import EmbeddingsConfig

logger = logging.getLogger(__name__)


def test_hydra_with_EmbeddingsConfig() -> None:
    with initialize_config_module(
        version_base=None,
        config_module="configs.embeddings",
    ):
        # config is relative to a module
        cfg: omegaconf.DictConfig = compose(
            config_name="basic_embeddings",
            overrides=[],
        )

        logger.info(f"cfg:\n{pprint.pformat(cfg)}")

        # This tests whether the configuration is valid
        config = EmbeddingsConfig.model_validate(
            obj=cfg,
        )

        logger.info(f"{type(config) = }")
        logger.info(f"config:\n{pprint.pformat(config)}")
