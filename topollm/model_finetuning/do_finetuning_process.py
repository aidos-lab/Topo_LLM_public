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
import os

import transformers

from topollm.config_classes.finetuning.FinetuningConfig import FinetuningConfig
from topollm.config_classes.MainConfig import MainConfig
from topollm.path_management.finetuning.FinetuningPathManagerFactory import (
    get_finetuning_path_manager,
)
from topollm.path_management.finetuning.FinetuningPathManagerProtocol import (
    FinetuningPathManager,
)
from topollm.data_handling.DatasetPreparerFactory import get_dataset_preparer
from topollm.model_finetuning.evaluate_tuned_model import evaluate_tuned_model
from topollm.model_finetuning.load_base_model import load_base_model
from topollm.model_finetuning.load_tokenizer import load_tokenizer

from topollm.model_finetuning.model_modifiers.ModelModifierFactory import (
    get_model_modifier,
)
from topollm.model_finetuning.prepare_model_input import prepare_model_input
from topollm.model_finetuning.save_tuned_model import save_tuned_model
from topollm.model_handling.get_torch_device import get_torch_device


def prepare_finetuned_model_dir(
    finetuning_path_manager: FinetuningPathManager,
    logger: logging.Logger = logging.getLogger(__name__),
) -> os.PathLike:
    finetuned_model_dir = finetuning_path_manager.finetuned_model_dir
    logger.info(f"{finetuned_model_dir = }")

    # Create the output directory if it does not exist
    finetuned_model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return finetuned_model_dir


def prepare_logging_dir(
    finetuning_path_manager: FinetuningPathManager,
    logger: logging.Logger = logging.getLogger(__name__),
) -> os.PathLike | None:
    logging_dir = finetuning_path_manager.logging_dir
    logger.info(f"{logging_dir = }")

    # Create the logging directory if it does not exist
    if logging_dir is not None:
        logging_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
    else:
        logger.info(
            f"No logging directory specified. "
            f"Using default logging from transformers.Trainer."
        )

    return logging_dir


def prepare_training_args(
    finetuning_config: FinetuningConfig,
    seed: int,
    finetuned_model_dir: os.PathLike,
    logging_dir: os.PathLike | None = None,
):
    training_args = transformers.TrainingArguments(
        output_dir=str(finetuned_model_dir),
        overwrite_output_dir=True,
        num_train_epochs=finetuning_config.num_train_epochs,
        max_steps=finetuning_config.max_steps,
        learning_rate=finetuning_config.learning_rate,
        weight_decay=finetuning_config.weight_decay,
        per_device_train_batch_size=finetuning_config.batch_sizes.train,
        per_device_eval_batch_size=finetuning_config.batch_sizes.eval,
        gradient_accumulation_steps=finetuning_config.gradient_accumulation_steps,
        gradient_checkpointing=finetuning_config.gradient_checkpointing,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
        fp16=finetuning_config.fp16,
        warmup_steps=finetuning_config.warmup_steps,
        evaluation_strategy="steps",
        eval_steps=finetuning_config.eval_steps,
        save_steps=finetuning_config.save_steps,
        logging_dir=logging_dir,  # type: ignore
        log_level=finetuning_config.log_level,
        logging_steps=finetuning_config.logging_steps,
        use_cpu=finetuning_config.use_cpu,
        seed=seed,
    )

    return training_args


def finetune_model(
    trainer: transformers.Trainer,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    logger.info(f"Calling trainer.train() ...")

    training_call_output = trainer.train(
        resume_from_checkpoint=False,
    )

    logger.info(f"Calling trainer.train() DONE")

    logger.info(f"training_call_output:\n" f"{training_call_output}")

    return None


def do_finetuning_process(
    main_config: MainConfig,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    finetuning_config = main_config.finetuning

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Use accelerator if available
    device = get_torch_device(
        preferred_torch_backend=main_config.preferred_torch_backend,
        logger=logger,
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Load data
    train_dataset_preparer = get_dataset_preparer(
        dataset_type=finetuning_config.finetuning_datasets.train_dataset.dataset_type,
        data_config=finetuning_config.finetuning_datasets.train_dataset,
        verbosity=main_config.verbosity,
        logger=logger,
    )
    train_dataset = train_dataset_preparer.prepare_dataset()

    eval_dataset_preparer = get_dataset_preparer(
        dataset_type=finetuning_config.finetuning_datasets.eval_dataset.dataset_type,
        data_config=finetuning_config.finetuning_datasets.eval_dataset,
        verbosity=main_config.verbosity,
        logger=logger,
    )
    eval_dataset = eval_dataset_preparer.prepare_dataset()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Load tokenizer and model

    tokenizer = load_tokenizer(
        finetuning_config=finetuning_config,
        logger=logger,
    )

    base_model = load_base_model(
        finetuning_config=finetuning_config,
        device=device,
        logger=logger,
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Potential modification of the model.
    # This allows for variations of the training process,
    # e.g. using LoRA or other model modifications.

    model_modifier = get_model_modifier(
        peft_config=finetuning_config.peft,
        device=device,
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

    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=finetuning_config.mlm_probability,
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Output paths

    finetuning_path_manager = get_finetuning_path_manager(
        config=main_config,
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
        train_dataset=train_dataset_mapped,  # type: ignore
        eval_dataset=eval_dataset_mapped,  # type: ignore
        tokenizer=tokenizer,
    )

    finetune_model(
        trainer=trainer,
        logger=logger,
    )

    save_tuned_model(
        trainer=trainer,
        logger=logger,
    )

    evaluate_tuned_model(
        trainer=trainer,
        logger=logger,
    )

    return None
