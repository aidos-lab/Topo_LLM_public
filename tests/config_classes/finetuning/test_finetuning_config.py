# Copyright 2024-2025
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

"""Test the FinetuningConfig class with Hydra."""

import logging
import pprint
from typing import TYPE_CHECKING

from hydra import compose, initialize_config_module

from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig

if TYPE_CHECKING:
    import omegaconf

logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def test_hydra_with_finetuning_config() -> None:
    """Test Hydra with FinetuningConfig."""
    with initialize_config_module(
        version_base=None,
        config_module="configs.finetuning",
    ):
        # config is relative to a module
        cfg: omegaconf.DictConfig = compose(
            config_name="finetuning_for_masked_lm",
            overrides=[
                "batch_sizes.eval=42",
            ],
        )

        logger.info(
            "cfg:\n%s",
            pprint.pformat(cfg),
        )

        # This tests whether the configuration is valid
        config = FinetuningConfig.model_validate(
            obj=cfg,
        )

        logger.info(
            f"{type(config) = }",  # noqa: G004 - low overhead
        )
        logger.info(
            "config:\n%s",
            pprint.pformat(config),
        )
