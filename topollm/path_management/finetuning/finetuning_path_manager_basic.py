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

"""Path manager for finetuning with basic functionality."""

import logging
import pathlib

from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.config_classes.data.data_config import DataConfig
from topollm.config_classes.finetuning.FinetuningConfig import FinetuningConfig
from topollm.config_classes.paths.paths_config import PathsConfig
from topollm.path_management.finetuning.peft.PEFTPathManagerFactory import (
    get_peft_path_manager,
)

logger = logging.getLogger(__name__)


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

        self.peft_path_manager = get_peft_path_manager(
            peft_config=self.finetuning_config.peft,
            verbosity=verbosity,
            logger=logger,
        )

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
            self.finetuning_config.base_model_config_description,
            self.peft_path_manager.peft_description_subdir,
            self.finetuning_parameters_description,
            self.training_progress_subdir,
        )

        return path

    @property
    def finetuning_parameters_description(
        self,
    ) -> str:
        desc = (
            f"{NAME_PREFIXES['learning_rate']}"
            f"{KV_SEP}"
            f"{self.finetuning_config.learning_rate}"
            f"{ITEM_SEP}"
            f"{NAME_PREFIXES['lr_scheduler_type']}"
            f"{KV_SEP}"
            f"{self.finetuning_config.lr_scheduler_type}"
            f"{ITEM_SEP}"
            f"{NAME_PREFIXES['weight_decay']}"
            f"{KV_SEP}"
            f"{self.finetuning_config.weight_decay}"
        )

        return desc

    @property
    def training_progress_subdir(
        self,
    ) -> pathlib.Path:
        path = pathlib.Path(
            self.epoch_description,
        )

        return path

    @property
    def epoch_description(
        self,
    ) -> str:
        desc = f"{NAME_PREFIXES['epoch']}{KV_SEP}{self.finetuning_config.num_train_epochs}"

        return desc

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
        """Return the logging directory for the finetuning run.

        We decide to return None here,
        because this will mean the logging_dir will be handled
        by the Trainer class.
        """
        path = None

        if self.verbosity >= 1:
            self.logger.info(f"logging_dir:\n" f"{path}")

        return path
