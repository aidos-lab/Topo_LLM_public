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

"""Perform the finetuning process."""

import logging
from typing import TYPE_CHECKING

import torch
import transformers

from topollm.config_classes.main_config import MainConfig
from topollm.data_handling.dataset_preparer.factory import get_dataset_preparer
from topollm.data_handling.dataset_preparer.protocol import DatasetPreparer
from topollm.data_handling.dataset_preparer.select_random_elements import (
    log_selected_dataset_elements_info,
)
from topollm.logging.log_dataset_info import log_huggingface_dataset_info
from topollm.model_finetuning.evaluate_tuned_model import evaluate_tuned_model
from topollm.model_finetuning.finetune_model import finetune_model
from topollm.model_finetuning.generate_from_pretrained_kwargs_instance import (
    extract_label_list,
    generate_from_pretrained_kwargs_instance,
)
from topollm.model_finetuning.get_compute_metrics import get_compute_metrics
from topollm.model_finetuning.gradient_modifiers.factory import get_gradient_modifier
from topollm.model_finetuning.load_base_model_from_finetuning_config import (
    load_base_model_from_finetuning_config,
)
from topollm.model_finetuning.load_tokenizer_from_finetuning_config import (
    load_tokenizer_from_finetuning_config,
)
from topollm.model_finetuning.model_modifiers.factory import (
    get_model_modifier,
)
from topollm.model_finetuning.prepare_data_collator import prepare_data_collator
from topollm.model_finetuning.prepare_finetuned_model_dir import (
    prepare_finetuned_model_dir,
)
from topollm.model_finetuning.prepare_logging_dir import prepare_logging_dir
from topollm.model_finetuning.prepare_model_input import prepare_model_input
from topollm.model_finetuning.prepare_training_args import prepare_training_args
from topollm.model_finetuning.save_tuned_model import save_tuned_model
from topollm.model_finetuning.trainer_modifiers.factory import get_trainer_modifier
from topollm.model_handling.model.token_classification_from_pretrained_kwargs import (
    TokenClassificationFromPretrainedKwargs,
)
from topollm.model_handling.tokenizer.tokenizer_modifier.factory import (
    get_tokenizer_modifier,
)
from topollm.model_handling.tokenizer.tokenizer_modifier.protocol import TokenizerModifier
from topollm.path_management.finetuning.factory import (
    get_finetuning_path_manager,
)
from topollm.typing.enums import Verbosity

if TYPE_CHECKING:
    import datasets

    from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig
    from topollm.model_finetuning.gradient_modifiers.protocol import GradientModifier
    from topollm.model_finetuning.model_modifiers.protocol import ModelModifier
    from topollm.model_finetuning.trainer_modifiers.protocol import TrainerModifier

default_device = torch.device("cpu")
default_logger = logging.getLogger(__name__)


def do_finetuning_process(
    main_config: MainConfig,
    device: torch.device = default_device,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Perform the finetuning process."""
    finetuning_config: FinetuningConfig = main_config.finetuning

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Load data
    train_dataset_preparer: DatasetPreparer = get_dataset_preparer(
        data_config=finetuning_config.finetuning_datasets.train_dataset,
        verbosity=main_config.verbosity,
        logger=logger,
    )
    train_dataset: datasets.Dataset = train_dataset_preparer.prepare_dataset()

    eval_dataset_preparer: DatasetPreparer = get_dataset_preparer(
        data_config=finetuning_config.finetuning_datasets.eval_dataset,
        verbosity=main_config.verbosity,
        logger=logger,
    )
    eval_dataset: datasets.Dataset = eval_dataset_preparer.prepare_dataset()

    # Print examples from the dataset
    if verbosity >= Verbosity.NORMAL:
        log_selected_dataset_elements_info(
            dataset=train_dataset,
            dataset_name="train_dataset",
            logger=logger,
        )
        log_selected_dataset_elements_info(
            dataset=eval_dataset,
            dataset_name="eval_dataset",
            logger=logger,
        )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Load tokenizer and model

    base_tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast = (
        load_tokenizer_from_finetuning_config(
            finetuning_config=finetuning_config,
            verbosity=verbosity,
            logger=logger,
        )
    )

    label_list: list[str] | None = extract_label_list(
        finetuning_config=finetuning_config,
        train_dataset=train_dataset,
        verbosity=verbosity,
        logger=logger,
    )

    from_pretrained_kwargs_instance: TokenClassificationFromPretrainedKwargs | None = (
        generate_from_pretrained_kwargs_instance(
            finetuning_config=finetuning_config,
            label_list=label_list,
        )
    )

    base_model: transformers.PreTrainedModel = load_base_model_from_finetuning_config(
        finetuning_config=finetuning_config,
        from_pretrained_kwargs_instance=from_pretrained_kwargs_instance,
        device=device,
        verbosity=verbosity,
        logger=logger,
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Potential modification of the tokenizer
    # (and the model if this is necessary for compatibility).
    # For instance, for some autoregressive models, the tokenizer
    # needs to be modified to add a padding token.

    tokenizer_modifier: TokenizerModifier = get_tokenizer_modifier(
        tokenizer_modifier_config=finetuning_config.base_model.tokenizer_modifier,
        verbosity=verbosity,
        logger=logger,
    )

    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast = (
        tokenizer_modifier.modify_tokenizer(
            tokenizer=base_tokenizer,
        )
    )
    base_model: transformers.PreTrainedModel = tokenizer_modifier.update_model(
        model=base_model,
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Potential modification of the model.
    # This allows for variations of the training process,
    # e.g. using LoRA or other model modifications.

    model_modifier: ModelModifier = get_model_modifier(
        peft_config=finetuning_config.peft,
        device=device,
        verbosity=verbosity,
        logger=logger,
    )
    modified_model = model_modifier.modify_model(
        model=base_model,
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Potential modification gradients.

    gradient_modifier: GradientModifier = get_gradient_modifier(
        gradient_modifier_config=finetuning_config.gradient_modifier,
        device=device,
        verbosity=verbosity,
        logger=logger,
    )
    gradient_modified_model = gradient_modifier.modify_gradients(
        model=modified_model,
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Prepare model input.

    train_dataset_mapped, eval_dataset_mapped = prepare_model_input(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        finetuning_config=finetuning_config,
    )

    if verbosity >= Verbosity.NORMAL:
        log_huggingface_dataset_info(
            dataset=train_dataset_mapped,
            dataset_name="train_dataset_mapped",
            logger=logger,
        )
        log_selected_dataset_elements_info(
            dataset=train_dataset_mapped,
            dataset_name="train_dataset_mapped",
            logger=logger,
        )

        log_huggingface_dataset_info(
            dataset=eval_dataset_mapped,
            dataset_name="eval_dataset_mapped",
            logger=logger,
        )
        log_selected_dataset_elements_info(
            dataset=eval_dataset_mapped,
            dataset_name="eval_dataset_mapped",
            logger=logger,
        )

    data_collator: transformers.DataCollatorForLanguageModeling | transformers.DataCollatorForTokenClassification = (
        prepare_data_collator(
            finetuning_config=finetuning_config,
            tokenizer=tokenizer,
            verbosity=verbosity,
            logger=logger,
        )
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Output paths

    finetuning_path_manager = get_finetuning_path_manager(
        main_config=main_config,
        logger=logger,
    )

    finetuned_model_dir = prepare_finetuned_model_dir(
        finetuning_path_manager=finetuning_path_manager,
        logger=logger,
    )

    logging_dir = prepare_logging_dir(
        finetuning_path_manager=finetuning_path_manager,
        logger=logger,
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Finetuning setup

    compute_metrics = get_compute_metrics(
        finetuning_config=finetuning_config,
        label_list=label_list,
        verbosity=verbosity,
        logger=logger,
    )

    training_args = prepare_training_args(
        finetuning_config=finetuning_config,
        seed=main_config.seed,
        finetuned_model_dir=finetuned_model_dir,
        logging_dir=logging_dir,
    )

    trainer: transformers.Trainer = transformers.Trainer(
        model=gradient_modified_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset_mapped,  # type: ignore - typing issue with Dataset
        eval_dataset=eval_dataset_mapped,  # type: ignore - typing issue with Dataset
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer_modifier: TrainerModifier = get_trainer_modifier(
        trainer_modifier_config=finetuning_config.trainer_modifier,
        verbosity=verbosity,
        logger=logger,
    )

    trainer: transformers.Trainer = trainer_modifier.modify_trainer(
        trainer=trainer,
    )

    finetune_model(
        trainer=trainer,
        verbosity=verbosity,
        logger=logger,
    )

    save_tuned_model(
        trainer=trainer,
        verbosity=verbosity,
        logger=logger,
    )

    evaluate_tuned_model(
        trainer=trainer,
        logger=logger,
    )
