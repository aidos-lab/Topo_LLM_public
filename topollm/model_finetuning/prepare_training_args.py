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

"""Prepare the training arguments for the finetuning process."""

import os

import transformers

from topollm.config_classes.finetuning.finetuning_config import FinetuningConfig


def prepare_training_args(
    finetuning_config: FinetuningConfig,
    seed: int,
    finetuned_model_dir: os.PathLike,
    logging_dir: os.PathLike | None = None,
) -> transformers.TrainingArguments:
    """Prepare the training arguments for the finetuning process."""
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
        evaluation_strategy="steps",
        eval_steps=finetuning_config.eval_steps,
        save_steps=finetuning_config.save_steps,
        logging_dir=logging_dir,  # type: ignore - typing problem with None and str
        log_level=finetuning_config.log_level,
        logging_steps=finetuning_config.logging_steps,
        use_cpu=finetuning_config.use_cpu,
        seed=seed,
    )

    return training_args
