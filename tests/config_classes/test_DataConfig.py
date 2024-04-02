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

import logging
import pprint

import pytest
from hydra import compose, initialize
import omegaconf

from topollm.config_classes.MainConfig import DataConfig

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "config_name",
    [
        "bbc",
        "iclr_2024_submissions",
        "multiwoz21",
        "multiwoz21_train",
        "multiwoz21_validation",
        "sgd",
        "wikitext",
    ],
)
def test_hydra_with_DataConfig(
    config_name: str,
) -> None:
    with initialize(
        config_path="../../configs/data/",
        version_base=None,
    ):
        # config is relative to a module
        cfg: omegaconf.DictConfig = compose(
            config_name=config_name,
            # overrides=[
            #     "dataset_description_string=overwritten_dataset_desc",
            # ],
        )

        logger.info(f"cfg:\n" f"{pprint.pformat(cfg)}")

        # This tests whether the configuration is valid
        config = DataConfig.model_validate(
            obj=cfg,
        )

        logger.info(f"{type(config) = }")
        logger.info(f"config:\n" f"{pprint.pformat(config)}")

    return None
