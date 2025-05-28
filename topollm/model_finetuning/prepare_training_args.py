# Copyright 2024
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


"""Prepare the training arguments for the finetuning process."""

import os

import transformers

from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig


def prepare_training_args(
    finetuning_config: FinetuningConfig,
    finetuned_model_dir: os.PathLike,
    logging_dir: os.PathLike | None = None,
) -> transformers.TrainingArguments:
    """Prepare the training arguments for the finetuning process."""
    # Note: the `label_names` argument appears to be necessary for the PEFT evaluation to work.
    # https://discuss.huggingface.co/t/eval-with-trainer-not-running-with-peft-lora-model/53286
    training_args = transformers.TrainingArguments(
        output_dir=str(finetuned_model_dir),
        overwrite_output_dir=True,
        num_train_epochs=finetuning_config.num_train_epochs,
        max_steps=finetuning_config.max_steps,
        learning_rate=finetuning_config.learning_rate,
        lr_scheduler_type=finetuning_config.lr_scheduler_type,
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
        eval_strategy="steps",
        eval_steps=finetuning_config.eval_steps,
        save_steps=finetuning_config.save_steps,
        label_names=[
            "labels",
        ],
        logging_dir=logging_dir,  # type: ignore - typing problem with None and str
        report_to=finetuning_config.report_to,
        log_level=finetuning_config.log_level,
        logging_steps=finetuning_config.logging_steps,
        use_cpu=finetuning_config.use_cpu,
        seed=finetuning_config.seed,
    )

    return training_args
