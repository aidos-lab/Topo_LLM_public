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
import pathlib

from topollm.config_classes.constants import NAME_PREFIXES
from topollm.config_classes.DataConfig import DataConfig
from topollm.config_classes.finetuning.FinetuningConfig import FinetuningConfig
from topollm.config_classes.PathsConfig import PathsConfig

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class FinetuningPathManagerBasic:
    def __init__(
        self,
        data_config: DataConfig,
        paths_config: PathsConfig,
        finetuning_config: FinetuningConfig,
        verbosity: int = 1,
        logger: logging.Logger = logging.getLogger(__name__),
    ):
        self.data_config = data_config
        self.finetuning_config = finetuning_config
        self.paths_config = paths_config

        self.verbosity = verbosity
        self.logger = logger

    @property
    def data_dir(
        self,
    ) -> pathlib.Path:
        return self.paths_config.data_dir

    @property
    def finetuned_base_dir(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.data_dir,
            "models",
            "finetuned_models",
            self.finetuning_config.finetuning_datasets.train_dataset.data_config_description,
            self.epoch_description,
        )

        return path

    @property
    def epoch_description(
        self,
    ):
        description = (
            f"{NAME_PREFIXES['epoch']}" f"{self.finetuning_config.num_train_epochs}"
        )

        return description

    @property
    def finetuned_model_dir(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.finetuned_base_dir,
            "model_files",
        )

        if self.verbosity >= 1:
            self.logger.info(f"finetuned_model_dir:\n" f"{path}")

        return path

    @property
    def logging_dir(
        self,
    ) -> pathlib.Path | None:
        """
        We decide to return None here,
        because this will mean the logging_dir will be handled
        by the Trainer class.
        """

        path = None

        if self.verbosity >= 1:
            self.logger.info(f"logging_dir:\n" f"{path}")

        return path
