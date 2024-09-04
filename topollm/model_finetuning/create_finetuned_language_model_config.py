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

"""Create the config for the language model resulting from fine-tuning."""

import logging
import pathlib
from typing import TYPE_CHECKING

import omegaconf

from topollm.config_classes.language_model.language_model_config import LanguageModelConfig
from topollm.config_classes.main_config import MainConfig
from topollm.model_finetuning.compute_last_save_step import compute_last_save_step
from topollm.path_management.finetuning.factory import get_finetuning_path_manager
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    from topollm.path_management.finetuning.protocol import FinetuningPathManager

default_logger = logging.getLogger(__name__)


def dump_language_model_config_to_file(
    language_model_config: LanguageModelConfig,
    configs_save_dir: pathlib.Path,
    config_file_name: str,
    *,
    generated_configs_logs_file_path: pathlib.Path | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Dump the language model config to a file."""
    if generated_configs_logs_file_path is None:
        generated_configs_logs_file_path = pathlib.Path(
            configs_save_dir,
            "generated_configs_logs",
            "generated_configs_logs.txt",
        )

    # Create the directories if they do not exist
    configs_save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    generated_configs_logs_file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_config_path = pathlib.Path(
        configs_save_dir,
        config_file_name,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "generated_config_path:\n%s",
            generated_config_path,
        )

    # Convert to yaml string
    #
    # Note: One needs to be careful in how the StrEnum instances will be serialized.
    # https://stackoverflow.com/questions/65209934/pydantic-enum-field-does-not-get-converted-to-string
    # For instance, language_model_config.model_dump() will return a dictionary with actual Enum instances,
    # for example:
    # ```
    # < language_model_config.model_dump(mode="json")
    # > {
    # >     'checkpoint_no': -1,
    # >     'lm_mode': <LMmode.MLM: 'mlm'>,
    # >     'task_type': <TaskType.MASKED_LM: 'masked_lm'>,
    # >     'pretrained_model_name_or_path': 'roberta-base',
    # >     'short_model_name': 'roberta-base',
    # >     'tokenizer_modifier': {'mode': <TokenizerModifierMode.DO_NOTHING: 'do_nothing'>, 'padding_token': '<pad>'}
    # > }
    # ```
    #
    # What we actually want is a dictionary with the string representation of the Enum instances,
    # which we can get by using the `mode="json"` argument of the `model_dump` method:
    # ```
    # < language_model_config.model_dump(mode="json")
    # > {
    # >     "checkpoint_no": -1,
    # >     "lm_mode": "mlm",
    # >     "task_type": "masked_lm",
    # >     "pretrained_model_name_or_path": "roberta-base",
    # >     "short_model_name": "roberta-base",
    # >     "tokenizer_modifier": {"mode": "do_nothing", "padding_token": "<pad>"},
    # > }
    # ```
    new_language_model_config_yaml_data: str = omegaconf.OmegaConf.to_yaml(
        language_model_config.model_dump(
            mode="json",
        ),
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "new_language_model_config_yaml_data:\n%s",
            new_language_model_config_yaml_data,
        )

    with generated_config_path.open(
        mode="w",
    ) as file:
        file.write(
            new_language_model_config_yaml_data,
        )

    # Append the name of the generated config to the logs file
    with generated_configs_logs_file_path.open(
        mode="a",
    ) as file:
        file.write(
            f"{config_file_name}\n",
        )


def update_language_model_config(
    base_language_model_config: LanguageModelConfig,
    finetuned_model_relative_dir: pathlib.Path,
    finetuned_short_model_name: str,
    checkpoint_no: int,
) -> LanguageModelConfig:
    """Update the language model config with the new finetuned model path and short model name."""
    new_pretrained_model_path = (
        r"${paths.data_dir}/" + str(finetuned_model_relative_dir) + r"/checkpoint-${language_model.checkpoint_no}"
    )

    new_short_model_name_with_checkpoint_interpolation = (
        str(finetuned_short_model_name) + r"_ckpt-${language_model.checkpoint_no}"
    )

    updated_config: LanguageModelConfig = base_language_model_config.model_copy(
        update={
            "pretrained_model_name_or_path": new_pretrained_model_path,
            "short_model_name": new_short_model_name_with_checkpoint_interpolation,
            "checkpoint_no": checkpoint_no,
        },
        deep=True,
    )

    return updated_config


def create_finetuned_language_model_config(
    main_config: MainConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Create the config for the language model resulting from fine-tuning.

    This config can be used for further processing, e.g. for the embedding and data generation.
    """
    finetuning_path_manager: FinetuningPathManager = get_finetuning_path_manager(
        main_config=main_config,
        logger=logger,
    )

    finetuned_model_relative_dir: pathlib.Path = finetuning_path_manager.get_finetuned_model_relative_dir()
    # The `finetuned_short_model_name` does not contain the checkpoint number appendix
    finetuned_short_model_name: str = finetuning_path_manager.get_finetuned_short_model_name()

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"{finetuned_model_relative_dir = }",  # noqa: G004 - low overhead
        )
        logger.info(
            f"{finetuned_short_model_name = }",  # noqa: G004 - low overhead
        )

    # # # #
    # Find the last checkpoint global save step
    finetuning_config = main_config.finetuning
    last_checkpoint_no = compute_last_save_step(
        total_samples=finetuning_config.finetuning_datasets.train_dataset.number_of_samples,
        batch_size=finetuning_config.batch_sizes.train,
        gradient_accumulation_steps=finetuning_config.gradient_accumulation_steps,
        num_epochs=finetuning_config.num_train_epochs,
        save_steps=finetuning_config.save_steps,
    )

    base_language_model_config: LanguageModelConfig = main_config.language_model
    new_language_model_config: LanguageModelConfig = update_language_model_config(
        base_language_model_config=base_language_model_config,
        finetuned_model_relative_dir=finetuned_model_relative_dir,
        finetuned_short_model_name=finetuned_short_model_name,
        checkpoint_no=last_checkpoint_no,
    )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "base_language_model_config:%s",
            base_language_model_config,
        )
        logger.info(
            "new_language_model_config:%s",
            new_language_model_config,
        )

    # # # #
    # Save the new config
    generated_configs_save_dir: pathlib.Path = pathlib.Path(
        main_config.paths.repository_base_path,
        "configs",
        "language_model",
    )

    dump_language_model_config_to_file(
        language_model_config=new_language_model_config,
        configs_save_dir=generated_configs_save_dir,
        config_file_name=f"{finetuned_short_model_name}.yaml",
        verbosity=verbosity,
        logger=logger,
    )
