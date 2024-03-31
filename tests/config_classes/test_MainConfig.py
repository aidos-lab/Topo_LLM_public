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

"""
Inspired by the examples from the hydra repository:
https://github.com/facebookresearch/hydra/blob/main/examples/advanced/hydra_app_example/tests/test_example.py
"""

import logging
import pprint

from hydra import compose, initialize, initialize_config_module
import omegaconf

from topollm.config_classes.MainConfig import MainConfig

logger = logging.getLogger(__name__)


def test_hydra_with_MainConfig() -> None:
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

        logger.info(f"cfg:\n" f"{pprint.pformat(cfg)}")

        # This tests whether the configuration is valid
        main_config = MainConfig.model_validate(
            obj=cfg,
        )

        logger.info(f"main_config:\n" f"{pprint.pformat(main_config)}")

    return None
