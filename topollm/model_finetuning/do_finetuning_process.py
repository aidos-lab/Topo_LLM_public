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

import torch
import transformers

from topollm.config_classes.main_config import MainConfig
from topollm.data_handling.dataset_preparer.factory import get_dataset_preparer
from topollm.model_finetuning.evaluate_tuned_model import evaluate_tuned_model
from topollm.model_finetuning.finetune_model import finetune_model
from topollm.model_finetuning.load_base_model_from_finetuning_config import (
    load_base_model_from_FinetuningConfig,
)
from topollm.model_finetuning.load_tokenizer_from_finetuning_config import (
    load_tokenizer_from_FinetuningConfig,
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
from topollm.model_handling.tokenizer.tokenizer_modifier.factory import (
    get_tokenizer_modifier,
)
from topollm.path_management.finetuning.factory import (
    get_finetuning_path_manager,
)

logger = logging.getLogger(__name__)


def do_finetuning_process(
    main_config: MainConfig,
    device: torch.device,
    verbosity: int = 1,
    logger: logging.Logger = logger,
) -> None:
    """Perform the finetuning process."""
    finetuning_config = main_config.finetuning

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Load data
    train_dataset_preparer = get_dataset_preparer(
        data_config=finetuning_config.finetuning_datasets.train_dataset,
        verbosity=main_config.verbosity,
        logger=logger,
    )
    train_dataset = train_dataset_preparer.prepare_dataset()

    eval_dataset_preparer = get_dataset_preparer(
        data_config=finetuning_config.finetuning_datasets.eval_dataset,
        verbosity=main_config.verbosity,
        logger=logger,
    )
    eval_dataset = eval_dataset_preparer.prepare_dataset()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Load tokenizer and model

    base_tokenizer = load_tokenizer_from_FinetuningConfig(
        finetuning_config=finetuning_config,
        verbosity=verbosity,
        logger=logger,
    )

    base_model = load_base_model_from_FinetuningConfig(
        finetuning_config=finetuning_config,
        device=device,
        verbosity=verbosity,
        logger=logger,
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Potential modification of the tokenizer
    # (and the model if this is necessary for compatibility).
    # For instance, for some autoregressive models, the tokenizer
    # needs to be modified to add a padding token.

    tokenizer_modifier = get_tokenizer_modifier(
        tokenizer_modifier_config=finetuning_config.tokenizer_modifier,
        verbosity=verbosity,
        logger=logger,
    )

    tokenizer = tokenizer_modifier.modify_tokenizer(
        tokenizer=base_tokenizer,
    )
    base_model = tokenizer_modifier.update_model(
        model=base_model,
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Potential modification of the model.
    # This allows for variations of the training process,
    # e.g. using LoRA or other model modifications.

    model_modifier = get_model_modifier(
        peft_config=finetuning_config.peft,
        device=device,
        verbosity=verbosity,
        logger=logger,
    )
    modified_model = model_modifier.modify_model(
        model=base_model,
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Prepare model input

    train_dataset_mapped, eval_dataset_mapped = prepare_model_input(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        finetuning_config=finetuning_config,
    )

    data_collator = prepare_data_collator(
        finetuning_config=finetuning_config,
        tokenizer=tokenizer,
        verbosity=verbosity,
        logger=logger,
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

    training_args = prepare_training_args(
        finetuning_config=finetuning_config,
        seed=main_config.seed,
        finetuned_model_dir=finetuned_model_dir,
        logging_dir=logging_dir,
    )

    trainer = transformers.Trainer(
        model=modified_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset_mapped,  # type: ignore - typing issue with Dataset
        eval_dataset=eval_dataset_mapped,  # type: ignore - typing issue with Dataset
        tokenizer=tokenizer,
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
