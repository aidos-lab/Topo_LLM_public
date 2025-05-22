# Copyright 2024
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

"""Test the DataConfig class."""

import logging
import pprint
from typing import TYPE_CHECKING

import pytest
from hydra import compose, initialize

from topollm.config_classes.main_config import DataConfig

if TYPE_CHECKING:
    import omegaconf

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


@pytest.mark.parametrize(
    "config_name",
    [
        "bbc",
        "iclr_2024_submissions",
        "multiwoz21",
        "multiwoz21_train",
        "multiwoz21_validation",
        "sgd",
        "wikitext-103-v1",
    ],
)
def test_hydra_with_data_config(
    config_name: str,
) -> None:
    """Test the DataConfig class with Hydra."""
    with initialize(
        # Note: `config_path` must be relative to the current file
        config_path="../../../configs/data/",  # Note: DO NOT change `config_path` to an absolute path
        version_base=None,
    ):
        # config is relative to a module
        cfg: omegaconf.DictConfig = compose(
            config_name=config_name,
        )

        default_logger.info(
            msg=f"cfg:\n{pprint.pformat(object=cfg)}",  # noqa: G004 - low overhead
        )

        # This tests whether the configuration is valid
        config: DataConfig = DataConfig.model_validate(
            obj=cfg,
        )

        default_logger.info(
            msg=f"{type(config) = }",  # noqa: G004 - low overhead
        )
        default_logger.info(
            msg=f"config:\n{pprint.pformat(config)}",  # noqa: G004 - low overhead
        )
