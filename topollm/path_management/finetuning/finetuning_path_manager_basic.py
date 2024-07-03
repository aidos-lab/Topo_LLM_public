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

    def get_finetuned_model_relative_dir(
        self,
    ) -> pathlib.Path:
        """Return the directory for the finetuned model relative to the data_dir."""
        path = pathlib.Path(
            "models",
            "finetuned_models",
            self.finetuning_config.finetuning_datasets.train_dataset.config_description,
            self.finetuning_config.base_model_config_description,
            self.peft_path_manager.peft_description_subdir,
            self.finetuning_config.gradient_modifier.gradient_modifier_description,
            self.finetuning_parameters_description,
            self.batch_size_description,
            self.training_progress_subdir,
            "model_files",
        )

        # TODO: Include the training objective here (i.e., masked lm or token classification)
        # TODO: Include the label name

        return path

    def get_finetuned_short_model_name(
        self,
    ) -> str:
        """Return the short model name for the finetuned model."""
        # Note: The short model name does not include the gradient modifier at the moment
        finetuned_short_model_name = (
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
                # Note: the short finetuning parameters description contains the epoch description
                self.get_finetuning_parameters_description(
                    description_type=DescriptionType.SHORT,
                ),
            )
        )

        return finetuned_short_model_name

    @property
    def batch_size_description(
        self,
    ) -> str:
        description = (
            f"{NAME_PREFIXES['batch_size_train']}" + f"{KV_SEP}" + f"{self.finetuning_config.batch_sizes.train}"
        )

        return description

    @property
    def finetuning_parameters_description(
        self,
    ) -> str:
        description = (
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

        return description

    def get_finetuning_parameters_description(
        self,
        description_type: DescriptionType = DescriptionType.LONG,
        short_description_separator: str = "-",
    ) -> str:
        """Return the config description."""
        match description_type:
            case DescriptionType.LONG:
                return self.finetuning_parameters_description
            case DescriptionType.SHORT:
                short_description = (
                    str(self.finetuning_config.learning_rate)
                    + short_description_separator
                    + self.finetuning_config.lr_scheduler_type
                    + short_description_separator
                    + str(self.finetuning_config.weight_decay)
                    + short_description_separator
                    + str(self.finetuning_config.num_train_epochs)
                )
                return short_description
            case _:
                msg = f"Unknown description type: {description_type}"
                raise ValueError(msg)

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
