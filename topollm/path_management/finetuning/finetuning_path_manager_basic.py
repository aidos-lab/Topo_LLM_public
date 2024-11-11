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
from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig
from topollm.config_classes.paths.paths_config import PathsConfig
from topollm.config_classes.sanitize_dirname import sanitize_dirname
from topollm.path_management.finetuning.peft.factory import (
    get_peft_path_manager,
)
from topollm.path_management.finetuning.peft.protocol import PEFTPathManager
from topollm.typing.enums import DescriptionType, Verbosity

default_logger = logging.getLogger(__name__)


class FinetuningPathManagerBasic:
    """Path manager for finetuning with basic functionality."""

    def __init__(
        self,
        data_config: DataConfig,
        paths_config: PathsConfig,
        finetuning_config: FinetuningConfig,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the path manager."""
        self.data_config: DataConfig = data_config
        self.finetuning_config: FinetuningConfig = finetuning_config
        self.paths_config: PathsConfig = paths_config

        self.peft_path_manager: PEFTPathManager = get_peft_path_manager(
            peft_config=self.finetuning_config.peft,
            verbosity=verbosity,
            logger=logger,
        )

        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

    @property
    def data_dir(
        self,
    ) -> pathlib.Path:
        return self.paths_config.data_dir

    def get_finetuned_model_relative_dir(
        self,
    ) -> pathlib.Path:
        """Return the directory for the finetuned model relative to the data_dir."""
        path = pathlib.Path(
            "models",
            "finetuned_models",
            self.finetuning_config.finetuning_datasets.train_dataset.long_config_description_with_data_splitting_without_data_subsampling,
            self.finetuning_config.base_model_config_description,
            self.peft_path_manager.peft_description_subdir,
            self.finetuning_config.gradient_modifier.gradient_modifier_description,
            self.finetuning_parameters_description_for_directory_partial_path,
            self.batch_size_description,
            self.training_duration_subdir,
            self.finetuning_reproducibility_description,
            "model_files",
        )

        return path

    def get_finetuned_short_model_name(
        self,
    ) -> str:
        """Return the short model name for the finetuned model."""
        # Note: The short model name does not include the gradient modifier at the moment
        finetuned_short_model_name: str = (
            str(self.finetuning_config.base_model_config_description)
            + ITEM_SEP
            + str(
                self.finetuning_config.finetuning_datasets.train_dataset.get_config_description(
                    description_type=DescriptionType.SHORT,
                ),
            )
            + ITEM_SEP
            + sanitize_dirname(str(self.peft_path_manager.peft_description_subdir))  # We might need to shorten this
            + ITEM_SEP
            + str(
                # Note: the short finetuning parameters description contains in particular:
                # - the finetuning seed
                # - the current epoch
                self.get_finetuning_parameters_description_for_short_model_name(),
            )
        )

        return finetuned_short_model_name

    @property
    def batch_size_description(
        self,
    ) -> str:
        description = (
            f"{NAME_PREFIXES['batch_size_train']}" + f"{KV_SEP}" + str(self.finetuning_config.batch_sizes.train)
        )

        return description

    @property
    def finetuning_parameters_description_for_directory_partial_path(
        self,
    ) -> str:
        description: str = (
            NAME_PREFIXES["learning_rate"]
            + KV_SEP
            + str(self.finetuning_config.learning_rate)
            + ITEM_SEP
            + NAME_PREFIXES["lr_scheduler_type"]
            + KV_SEP
            + str(self.finetuning_config.lr_scheduler_type)
            + ITEM_SEP
            + NAME_PREFIXES["weight_decay"]
            + KV_SEP
            + str(self.finetuning_config.weight_decay)
        )

        return description

    @property
    def finetuning_reproducibility_description(
        self,
    ) -> str:
        description: str = NAME_PREFIXES["seed"] + KV_SEP + str(self.finetuning_config.seed)

        return description

    def get_finetuning_parameters_description_for_short_model_name(
        self,
        short_description_separator: str = "-",
    ) -> str:
        """Return the config description."""
        short_description = (
            str(self.finetuning_config.learning_rate)
            + short_description_separator
            + str(self.finetuning_config.lr_scheduler_type)
            + short_description_separator
            + str(self.finetuning_config.weight_decay)
            + short_description_separator
            + str(self.finetuning_config.num_train_epochs)
        )
        return short_description

    @property
    def training_duration_subdir(
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
        description = f"{NAME_PREFIXES['epoch']}{KV_SEP}{self.finetuning_config.num_train_epochs}"

        return description

    @property
    def finetuned_model_dir(
        self,
    ) -> pathlib.Path:
        """Absolute path to the directory for the finetuned model."""
        path = pathlib.Path(
            self.data_dir,
            self.get_finetuned_model_relative_dir(),
        )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                "finetuned_model_dir:\n%s",
                path,
            )

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
            self.logger.info(
                "logging_dir:\n%s",
                path,
            )

        return path
