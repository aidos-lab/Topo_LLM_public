# coding=utf-8
#
# Copyright 2023 Heinrich Heine University Duesseldorf
#
# Authors: Benjamin Matthias Ruppik
# This code was generated with the help of AI writing assistants
# including GitHub Copilot, ChatGPT and Bing Chat.
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
Script for fine-tuning language models on dialogue data.
"""

import argparse
import json
import logging
import math
import os
import pathlib
import sys
from datetime import datetime

import datasets
import torch
import transformers
from convlab.util import load_dataset, load_ontology
from datasets import Dataset
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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Configure the logging module

# Get the current date and time
now = datetime.now()

# Format the date and time as a string
# This will produce a string like "20230519-172530" for May 19, 2023 at 17:25:30
timestamp = now.strftime("%Y%m%d-%H%M%S")

logfile_path = pathlib.Path(
    "logs",
    f"finetune_masked_language_model_with_transformers_on_dialogue_data_{timestamp}.log",
)

logging.basicConfig(
    # Set the logging level
    # level=logging.DEBUG,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",  # Set the logging format
    handlers=[  # Set the logging handlers
        logging.FileHandler(logfile_path),  # Log to a file
        logging.StreamHandler(sys.stdout),  # Log to the console
    ],
    force=True,  # make logging work in jupyter notebook
)

# Get a logger object
logger = logging.getLogger(__name__)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Ignore KeyboardInterrupt, just re-raise
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    else:
        # Log the exception
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


# Set the function to handle unhandled exceptions
sys.excepthook = handle_exception

logger.info(f"Logging to file '{logfile_path}' and stdout")

# END Configure the logging module
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals


# Get the base path from the environment variable
term_extraction_base_path_env = os.environ.get("TERM_EXTRACTION_BASE_PATH")
if term_extraction_base_path_env is None:
    raise ValueError("Environment variable 'TERM_EXTRACTION_BASE_PATH' is not set")
else:
    term_extraction_base_path_env = os.path.expandvars(
        term_extraction_base_path_env
    )  # Replace the $HOME part with the user's home directory
    term_extraction_base_path = pathlib.Path(
        term_extraction_base_path_env
    ).resolve()  # Compute the canonical, absolute form
    logger.info(f"term_extraction_base_path: {term_extraction_base_path}")

tda_base_path_env = os.environ.get("TDA_BASE_PATH")
if tda_base_path_env is None:
    raise ValueError("Environment variable 'TDA_BASE_PATH' is not set")
else:
    tda_base_path_env = os.path.expandvars(tda_base_path_env)  # Replace the $HOME part with the user's home directory
    tda_base_path = pathlib.Path(tda_base_path_env).resolve()  # Compute the canonical, absolute form
    logger.info(f"tda_base_path: {tda_base_path}")

# # check if the paths are valid directories
# if not term_extraction_base_path.is_dir():
#     raise ValueError(
#         f"term_extraction_base_path '{term_extraction_base_path}' "
#         f"is not a directory"
#     )
# if not tda_base_path.is_dir():
#     raise ValueError(f"tda_base_path '{tda_base_path}' " f"is not a directory")

tda_base_path = "./"

# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def parse_args():
    """
    The paths should be given relative to the environment variables
    TERM_EXTRACTION_BASE_PATH and TDA_BASE_PATH.
    """

    parser = argparse.ArgumentParser()

    # # # #
    # Required parameters

    # # # #
    # Optional parameters

    parser.add_argument(
        "--max_length",
        default=512,
        type=int,
        required=False,
        help="Maximum length of model sequence.",
    )

    parser.add_argument(
        "--train_batch_size",
        default=8,
        type=int,
        required=False,
        help="Batch size for training.",
    )

    parser.add_argument(
        "--eval_batch_size",
        default=16,
        type=int,
        required=False,
        help="Batch size for training.",
    )

    # add the data-set desc list as arguments which should be a list of strings
    parser.add_argument(
        "--data_set_desc_list",
        default=None,
        required=True,
        help="List of data-set descriptions.",
        nargs="+",  # this is to make sure that the list is parsed correctly
    )

    parser.add_argument(
        "--debug_index",
        default=None,
        type=int,
        required=False,
        help="Cut off the dataset here for debugging purposes.",
    )

    parser.add_argument(
        "--context",
        default="dialogue",
        type=str,
        required=False,
        help="Context of the input data.",
    )

    args = parser.parse_args()

    return args


def main():
    # # # # # # # # # # # # # # # # # # # # # # # #
    # Parse arguments
    args = parse_args()

    # Convert the Namespace object to a dictionary
    args_dict = vars(args)
    # Pretty print the dictionary
    pretty_args = json.dumps(
        args_dict,
        indent=4,
    )
    logger.info(f"args:\n{pretty_args}")

    train_batch_size: int = args.train_batch_size
    logger.info(f"train_batch_size: {train_batch_size}")

    eval_batch_size: int = args.eval_batch_size
    logger.info(f"eval_batch_size: {eval_batch_size}")

    debug_index: int | None = args.debug_index
    logger.info(f"debug_index: {debug_index}")

    # context = "utterance"
    # context = "dialogue"
    context = args.context

    model_identifier = "roberta-base"
    tokenizer_identifier = "roberta-base"

    max_length: int = args.max_length
    logging.info(f"max_length: {max_length}")

    dataset_desc_list = args.data_set_desc_list
    logger.info(f"dataset_desc_list: {dataset_desc_list}")

    # Set the transformers logging level
    transformers.logging.set_verbosity_info()

    # # # #
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Load data

    logging.info(f"Loading datasets ...")

    # create the dataset string
    dataset_string = "_".join(dataset_desc_list)

    # handle conll2003 dataset that is not in convlab but only in general huggingface datasets
    conll2003 = False
    if "conll2003" in dataset_desc_list:
        dataset_desc_list.remove("conll2003")
        conll2003 = True

    datasets_dict = {
        dataset_desc: load_dataset(
            dataset_name=dataset_desc,
        )
        for dataset_desc in dataset_desc_list
    }

    # add conll2003 dataset if it was in the list
    if conll2003:
        conll = datasets.load_dataset("conll2003")
        datasets_dict["conll2003"] = conll

    # ontology functionality is not necessary for our application
    # ontology = load_ontology(embeddings_config["data"]["dataset_desc"])
    # database functionality is not necessary for our application
    # database = load_database(embeddings_config['data']['dataset_desc'])

    logging.info(f"Loading datasets DONE")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Load tokenizer and model

    logger.info(f"Loading tokenizer: {tokenizer_identifier} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_identifier,
    )
    logger.info(f"Loading tokenizer: {tokenizer_identifier} DONE")

    # Set the padding token to the eos token
    tokenizer.pad_token = tokenizer.eos_token
    logging.info(f"tokenizer:\n{tokenizer}")

    logging.info(f"Loading model: {model_identifier} ...")
    model = AutoModelForMaskedLM.from_pretrained(
        model_identifier,
    )
    logging.info(f"Loading model: {model_identifier} DONE")

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
    # Create datasets for finetuning and saving files

    train_dataset_path = pathlib.Path(
        tda_base_path,
        "contextual_dialogues",
        "data",
        "dialogue_data_text_files",
        f"dataset_{dataset_string}_train_context_{context}_debug_idx_{debug_index}.txt",
    )
    logging.info(f"train_dataset_path: {train_dataset_path} ...")

    # create directory if it does not exist
    logging.info(f"Creating directory: {train_dataset_path.parent} ...")
    train_dataset_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_dataset_path = pathlib.Path(
        tda_base_path,
        "contextual_dialogues",
        "data",
        "dialogue_data_text_files",
        f"dataset_{dataset_string}_validation_context_{context}_debug_idx_{debug_index}.txt",
    )
    logging.info(f"validation_dataset_path: {validation_dataset_path} ...")

    # create directory if it does not exist
    logging.info(f"Creating directory: {validation_dataset_path.parent} ...")
    validation_dataset_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # # # #
    # Create the datasets from the dialogue utterances

    # Create and open the files for writing
    logging.info(f"Creating files for writing ...")
    train_file = open(
        train_dataset_path,
        "w",
    )
    validation_file = open(
        validation_dataset_path,
        "w",
    )

    logging.info(f"Creating datasets from the dialogue utterances ...")

    # Example: access specific utterance
    # debug_utterance = dataset["train"][14]["turns"][1]["utterance"]

    if context == "utterance":
        logging.info(f"context: {context}")
        for dataset_desc, dataset in tqdm(datasets_dict.items()):
            logger.info(f"dataset_desc: {dataset_desc}")
            # Iterate over the training dialogues
            logger.info(f"len(dataset['train']): {len(dataset['train'])}")

            if dataset_desc == "conll2003":
                for row in tqdm(
                    dataset["train"][:debug_index] if debug_index else dataset["train"],
                    desc=f"dataset_desc: {dataset_desc} train",
                ):
                    utterance = " ".join(row["tokens"])
                    train_file.write(utterance + "\n")

                # Iterate over the validation sentences
                logger.info(f"len(dataset['validation']): {len(dataset['validation'])}")
                for row in tqdm(
                    (dataset["validation"][:debug_index] if debug_index else dataset["validation"]),
                    desc=f"dataset_desc: {dataset_desc} validation",
                ):
                    utterance = " ".join(row["tokens"])
                    validation_file.write(utterance + "\n")

            else:
                for dialogue_dict in tqdm(
                    dataset["train"][:debug_index] if debug_index else dataset["train"],
                    desc=f"dataset_desc: {dataset_desc} train",
                ):
                    for turn_dict in dialogue_dict["turns"]:
                        utterance = turn_dict["utterance"]

                        # Write the utterance to the file
                        train_file.write(utterance + "\n")

                # Iterate over the validation dialogues
                logger.info(f"len(dataset['validation']): {len(dataset['validation'])}")

                for dialogue_dict in tqdm(
                    (dataset["validation"][:debug_index] if debug_index else dataset["validation"]),
                    desc=f"dataset_desc: {dataset_desc} validation",
                ):
                    for turn_dict in dialogue_dict["turns"]:
                        utterance = turn_dict["utterance"]

                        # Write the utterance to the file
                        validation_file.write(utterance + "\n")

    elif context == "dialogue":
        if dataset_desc == "conll2003":
            logger.info("for conll2003 dataset, context cannot be dialogue")
            raise ValueError("for conll2003 dataset, context cannot be dialogue")

        else:
            for dataset_desc, dataset in tqdm(datasets_dict.items()):
                logger.info(f"dataset_desc: {dataset_desc}")

                # Iterate over the training dialogues
                logger.info(f"len(dataset['train']): {len(dataset['train'])}")

                for dialogue_dict in tqdm(
                    dataset["train"][:debug_index] if debug_index else dataset["train"],
                    desc=f"dataset_desc: {dataset_desc} train",
                ):
                    current_dialogue = ""

                    for turn_dict in dialogue_dict["turns"]:
                        utterance = turn_dict["utterance"]

                        # Concatenate the utterance to the current dialogue in the same line
                        current_dialogue += utterance + " "

                    # Write the current dialogue to the file
                    train_file.write(current_dialogue + "\n")

                # Iterate over the validation dialogues
                logger.info(f"len(dataset['validation']): {len(dataset['validation'])}")

                for dialogue_dict in tqdm(
                    (dataset["validation"][:debug_index] if debug_index else dataset["validation"]),
                    desc=f"dataset_desc: {dataset_desc} validation",
                ):
                    current_dialogue = ""

                    for turn_dict in dialogue_dict["turns"]:
                        utterance = turn_dict["utterance"]

                        # Concatenate the utterance to the current dialogue in the same line
                        current_dialogue += utterance + " "

                    # Write the current dialogue to the file
                    validation_file.write(current_dialogue + "\n")
    else:
        raise ValueError(f"Unknown context '{context}'")

    logging.info(f"Creating datasets from the dialogue utterances DONE")

    # Close the files
    logging.info(f"Closing files ...")
    train_file.close()
    validation_file.close()
    logging.info(f"Closing files DONE")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #

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
    logging.info(f"finetuned_model_output_dir:\n{finetuned_model_output_dir}")

    # Create the output directory if it does not exist
    logging.info(f"Creating finetuned_model_output_dir:\n{finetuned_model_output_dir}\n...")
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
    logging.info(f"Creating logging_directory:\n{logging_directory}\n...")
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

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # OPTIONAL: Test the model

    finetuned_pipeline = pipeline(
        "fill-mask",
        model=str(finetuned_model_output_dir),
        tokenizer=tokenizer_identifier,
    )
    prompt = "I am looking for a <mask>"
    logger.info(f"prompt:\n{prompt}")

    result = finetuned_pipeline(prompt)
    logger.info(f"result:\n{result}")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    logger.info(f"================================")
    logger.info(f"==== Finetuning script DONE ====")
    logger.info(f"================================")

    sys.exit(0)


if __name__ == "__main__":
    main()
