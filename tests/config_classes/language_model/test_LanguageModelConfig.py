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
from hydra import compose, initialize_config_module
import omegaconf

from topollm.config_classes.LanguageModelConfig import LanguageModelConfig

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "config_name",
    [
        "roberta-base",
        "roberta-base_finetuned-on-multiwoz21-train-5000_context-utterance_ep-3",
        "roberta-base_finetuned-on-multiwoz21-train-5000_context-utterance_ep-5",
        "roberta-base_finetuned-on-multiwoz21-train_context-dialogue",
        "roberta-base_finetuned-on-multiwoz21-train-and-sgd-train_context-dialogue",
    ],
)
def test_hydra_with_LanguageModelConfig(
    config_name: str,
) -> None:
    with initialize_config_module(
        version_base=None,
        config_module="configs.language_model",
    ):
        # config is relative to a module
        cfg: omegaconf.DictConfig = compose(
            config_name=config_name,
            overrides=[
                "pretrained_model_name_or_path=overridden_pretrained_model_name_or_path",
                "short_model_name=overridden_short_model_name",
            ],
        )

        logger.info(f"cfg:\n" f"{pprint.pformat(cfg)}")

        # This tests whether the configuration is valid
        config = LanguageModelConfig.model_validate(
            obj=cfg,
        )

        logger.info(f"{type(config) = }")
        logger.info(f"config:\n" f"{pprint.pformat(config)}")

    return None
