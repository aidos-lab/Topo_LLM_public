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

"""
Script for fine-tuning language model on huggingface datasets.
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# System imports
import argparse
import json
import logging
import math
import os
import pathlib
import sys
from datetime import datetime

# Third party imports
import hydra
import hydra.core.hydra_config
import omegaconf
import datasets
import numpy as np
import pandas as pd
import torch
import transformers
import datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    pipeline,
)

# Local imports
from topollm.data_handling.DatasetPreparerFactory import get_dataset_preparer
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.config_classes.MainConfig import MainConfig
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_handling.load_tokenizer import load_tokenizer
from topollm.model_handling.load_model import load_model
from topollm.model_handling.get_torch_device import get_torch_device

# END Imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

# A logger for this file
global_logger = logging.getLogger(__name__)

setup_exception_logging(
    logger=global_logger,
)

# torch.set_num_threads(1)

# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@hydra.main(
    config_path="../../configs",
    config_name="main_config",
    version_base="1.2",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    """Run the script."""

    global_logger.info("Running script ...")

    # Set the transformers logging level
    transformers.logging.set_verbosity_info()

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=global_logger,
    )

    run_finetuning(
        main_config=main_config,
        logger=global_logger,
    )

    return None


def run_finetuning(
    main_config: MainConfig,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    finetuning_config = main_config.finetuning

    # # # #
    # Use accelerator if available
    device = get_torch_device(
        preferred_torch_backend=main_config.preferred_torch_backend,
        logger=global_logger,
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
    
    # TODO Continue here
    


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Load tokenizer and model

    # ? TODO Do we need to add parameters to the tokenizer here for finetuning?

    logger.info(f"Loading tokenizer " f"{finetuning_config.pretrained_model_name_or_path = } ...")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=finetuning_config.pretrained_model_name_or_path,
    )
    logger.info(f"Loading tokenizer " f"{finetuning_config.pretrained_model_name_or_path = } DONE")

    # Set the padding token to the eos token
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"tokenizer:\n{tokenizer}")

    logging.info(f"Loading model " f"{model_identifier} ...")
    model = AutoModelForMaskedLM.from_pretrained(
        pretrained_model_name_or_path=finetuning_config.pretrained_model_name_or_path,
    )
    logging.info(f"Loading model " f"{model_identifier} DONE")

    logging.info(f"model:\n{model}")
    logging.info(f"model.config:\n{model.config}")

    # Check type of model
    assert isinstance(
        model,
        PreTrainedModel,
    )

    # Move the model to GPU if available
    logging.info(f"Moving model to device: {device} ...")
    model.to(device)  # type: ignore
    logging.info(f"Moving model to device: {device} DONE")

 


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Tokenize data using the Dataset.map() function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    # Preparing custom datasets
    train_dataset = Dataset.from_dict(
        {
            "text": open(str(train_dataset_path)).readlines(),
        }
    )
    validation_dataset = Dataset.from_dict(
        {
            "text": open(str(validation_dataset_path)).readlines(),
        }
    )

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
    )
    validation_dataset = validation_dataset.map(
        tokenize_function,
        batched=True,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    finetuned_model_output_dir = pathlib.Path(
        tda_base_path,
        "contextual_dialogues",
        "data",
        "models",
        f"roberta-base_finetuned_on_{dataset_string}_train_context_{context}_debug_idx_{debug_index}",
    )
    logging.info(f"finetuned_model_output_dir:\n" f"{finetuned_model_output_dir}")

    # Create the output directory if it does not exist
    logging.info(
        f"Creating finetuned_model_output_dir:\n{finetuned_model_output_dir}\n..."
    )
    finetuned_model_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logging_directory = pathlib.Path(
        tda_base_path,
        "contextual_dialogues",
        "data",
        "logs",
        f"roberta-base_finetuned_on_{dataset_string}_train_context_{context}_debug_idx_{debug_index}",
    )

    # Create the output directory if it does not exist
    logging.info(f"Creating logging_directory:\n" f"{logging_directory}\n...")
    logging_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_args = TrainingArguments(
        output_dir=str(finetuned_model_output_dir),  # The output directory
        overwrite_output_dir=True,  # overwrite the content of the output directory
        num_train_epochs=5,  # number of training epochs
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=train_batch_size,  # batch size for training
        per_device_eval_batch_size=eval_batch_size,  # batch size for evaluation
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        fp16=True,
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        evaluation_strategy="steps",
        eval_steps=400,  # Number of update steps between two evaluations.
        save_steps=800,  # after # steps model is saved
        logging_dir=str(logging_directory),
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Train the model

    logger.info(f"Calling trainer.train() ...")

    # Train the model
    #
    training_call_output = trainer.train(
        resume_from_checkpoint=False,
    )

    # Train the model when resuming from a checkpoint
    #
    # training_call_output = trainer.train(
    #     resume_from_checkpoint="data/models/roberta-base_finetuned_on_{dataset_string}_train_context_dialogue_debug_idx_None/checkpoint-3200",
    # )

    logger.info(f"Calling trainer.train() DONE")

    logger.info(f"training_call_output:\n{training_call_output}")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Save the model

    logger.info(f"Calling trainer.save_model() ...")
    trainer.save_model()
    logger.info(f"Calling trainer.save_model() DONE")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Evaluate the model

    logger.info(f"Evaluating the model ...")
    eval_results = trainer.evaluate()
    logger.info(f"Perplexity:\n{math.exp(eval_results['eval_loss']):.2f}")
    logger.info(f"eval_results:\n{eval_results}")

    return None


if __name__ == "__main__":
    main()
