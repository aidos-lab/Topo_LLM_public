# Copyright 2024-2025
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


"""Path manager for finetuning with basic functionality."""

import logging
import pathlib
from typing import TYPE_CHECKING

from topollm.config_classes.constants import ITEM_SEP, KV_SEP, NAME_PREFIXES
from topollm.config_classes.data.data_config import DataConfig
from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig
from topollm.config_classes.paths.paths_config import PathsConfig
from topollm.path_management.finetuning.peft.factory import (
    get_peft_path_manager,
)
from topollm.typing.enums import DescriptionType, Verbosity

if TYPE_CHECKING:
    from topollm.path_management.finetuning.peft.protocol import PEFTPathManager

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


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
        # Notes:
        # - The tokenizer.config_description should be part of the fine-tuned model directory,
        #   because this determines the processing of the input data in the fine-tuning process.
        path = pathlib.Path(
            "models",
            "finetuned_models",
            self.finetuning_config.finetuning_datasets.train_dataset.get_partial_path(),
            self.finetuning_config.tokenizer.get_config_description(
                description_type=DescriptionType.LONG,
            ),
            self.finetuning_config.get_base_model_config_description(
                description_type=DescriptionType.LONG,
            ),
            self.peft_path_manager.peft_description_subdir,
            self.finetuning_config.gradient_modifier.get_config_description(
                description_type=DescriptionType.LONG,
            ),
            self.finetuning_parameters_partial_path,
            self.batch_size_description,
            self.training_duration_subdir,
            self.finetuning_reproducibility_description,
            "model_files",
        )

        return path

    def get_finetuned_short_model_name(
        self,
        short_description_separator: str = "-",
    ) -> str:
        """Return the short model name for the finetuned model.

        Note: Since the short model name is used to identify the language model config file,
        it should NOT include the equals sign "=",
        since this will interfer with the overwrite syntax for the hydra config command line interface.

        Note:
        - Dropout parameters are part of the base model config description.

        """
        finetuned_short_model_name: str = (
            str(
                object=self.finetuning_config.get_base_model_config_description(
                    description_type=DescriptionType.SHORT,
                    short_description_separator=short_description_separator,
                ),
            )
            + ITEM_SEP
            + str(
                object=self.finetuning_config.finetuning_datasets.train_dataset.get_config_description(
                    description_type=DescriptionType.SHORT,
                    short_description_separator=short_description_separator,
                ),
            )
            + ITEM_SEP
            + str(
                object=self.finetuning_config.tokenizer.get_config_description(
                    description_type=DescriptionType.SHORT,
                    short_description_separator=short_description_separator,
                ),
            )
            + ITEM_SEP
            + self.peft_path_manager.get_config_description(
                description_type=DescriptionType.SHORT,
                short_description_separator=short_description_separator,
            )
            + ITEM_SEP
            + str(
                # Note: the short finetuning parameters description does NOT contain:
                # - the finetuning seed
                # - the current epoch
                # We handle these through value interpolation via the hydra config system.
                object=self.get_finetuning_parameters_description_for_short_model_name(
                    short_description_separator=short_description_separator,
                ),
            )
        )

        return finetuned_short_model_name

    @property
    def batch_size_description(
        self,
    ) -> str:
        description = (
            f"{NAME_PREFIXES['batch_size_train']}" + f"{KV_SEP}" + str(object=self.finetuning_config.batch_sizes.train)
        )

        return description

    @property
    def finetuning_parameters_partial_path(
        self,
    ) -> str:
        """Return the partial path for the finetuning parameters.

        Note:
        - Dropout parameters are part of the base model config description.

        """
        learning_parameters_description: str = (
            NAME_PREFIXES["learning_rate"]
            + KV_SEP
            + str(object=self.finetuning_config.learning_rate)
            + ITEM_SEP
            + NAME_PREFIXES["lr_scheduler_type"]
            + KV_SEP
            + str(object=self.finetuning_config.lr_scheduler_type)
            + ITEM_SEP
            + NAME_PREFIXES["weight_decay"]
            + KV_SEP
            + str(object=self.finetuning_config.weight_decay)
        )

        return learning_parameters_description

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
        """Return the config description.

        Note:
        - Dropout parameters are part of the base model config description.

        """
        short_description: str = (
            str(object=self.finetuning_config.learning_rate)
            + short_description_separator
            + str(object=self.finetuning_config.lr_scheduler_type)
            + short_description_separator
            + str(object=self.finetuning_config.weight_decay)
            + short_description_separator
            + self.finetuning_config.gradient_modifier.get_config_description(
                description_type=DescriptionType.SHORT,
                short_description_separator=short_description_separator,
            )
            + short_description_separator
            + str(object=self.finetuning_config.num_train_epochs)
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
        description: str = f"{NAME_PREFIXES['epoch']}{KV_SEP}{self.finetuning_config.num_train_epochs}"

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
