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

"""Test the MainConfig class.

Inspired by the examples from the hydra repository:
https://github.com/facebookresearch/hydra/blob/main/examples/advanced/hydra_app_example/tests/test_example.py
"""

import logging
import pprint
from typing import TYPE_CHECKING

from hydra import compose, initialize

from topollm.config_classes.main_config import MainConfig

if TYPE_CHECKING:
    import omegaconf

logger = logging.getLogger(__name__)


def test_hydra_with_main_config() -> None:
    """Test the MainConfig class with Hydra."""
    with initialize(
        config_path="../../configs",
        version_base=None,
    ):
        # config is relative to a module
        cfg: omegaconf.DictConfig = compose(
            config_name="main_config",
            overrides=[
                "data.number_of_samples=6000",
            ],
        )

        logger.info(
            msg=f"cfg:\n{pprint.pformat(object=cfg)}",  # noqa: G004 - low overhead
        )

        # This tests whether the configuration is valid
        config: MainConfig = MainConfig.model_validate(
            obj=cfg,
        )

        logger.info(
            msg=f"{type(config) = }",  # noqa: G004 - low overhead
        )
        logger.info(
            msg=f"config:\n{pprint.pformat(object=config)}",  # noqa: G004 - low overhead
        )
