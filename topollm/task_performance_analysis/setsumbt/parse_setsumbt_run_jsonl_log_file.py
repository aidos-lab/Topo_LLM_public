# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (mail@ruppik.net)
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

"""Parse a SetSUMBT JSONL log file."""

import json
import logging
import os
import pathlib
from typing import Any

import pandas as pd

from topollm.logging.log_dataframe_info import log_dataframe_info
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def parse_setsumbt_run_jsonl_log_file(
    file_path: os.PathLike,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> pd.DataFrame:
    """Parse a SETSUMBT JSONL log file containing training checkpoints and evaluation statistics.

    The log file is expected to contain JSON objects on each line. The function looks for
    two types of log entries:
      1. A checkpoint entry with a message like "Step <number> complete" and a key
         "Loss since last update". This entry gives the checkpoint (step) and loss.
      2. An evaluation entry with the message "Validation set statistics" that includes
         various validation metrics (e.g., "Joint Goal Accuracy", "Slot F1 Score", etc.).

    The function pairs each checkpoint entry with the following validation entry and returns
    a DataFrame where each row corresponds to a checkpoint with its associated loss and validation
    metrics.

    Args:
        file_path:
            The path to the JSONL log file.
        verbosity:
            The verbosity level for logging and output.
        logger:
            The logger to use for output.


    Returns:
        pd.DataFrame: A DataFrame with columns for checkpoint number, loss, timestamps, and
                      validation metrics.

    """
    results: list[dict[str, Any]] = []
    current_checkpoint: dict[str, Any] = {}

    # Use Path to allow easy file handling
    log_file = pathlib.Path(
        file_path,
    )
    if not log_file.is_file():
        msg: str = f"File not found: {file_path = }"
        raise FileNotFoundError(
            msg,
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Reading {log_file = } ...",  # noqa: G004 - low overhead
        )

    # Open the file and process line by line
    with log_file.open(
        mode="r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line: str = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                log_entry = json.loads(
                    s=line,
                )
            except json.JSONDecodeError as e:
                # If a line is not valid JSON, skip it or handle as needed.
                logger.info(
                    msg=f"Skipping invalid JSON line: {line}\nError: {e}",  # noqa: G004 - low overhead
                )
                continue

            message: str = log_entry.get("message", "")

            # Look for checkpoint entries: message starts with "Step" and contains "complete"
            # and the entry includes a "Loss since last update"
            if message.startswith("Step") and "complete" in message and "Loss since last update" in log_entry:
                # Extract the step number from the message, e.g., "Step 2813 complete"
                parts: list[str] = message.split()
                try:
                    step_number = int(parts[1])
                except (IndexError, ValueError):
                    step_number = None

                # Build a dictionary for this checkpoint
                current_checkpoint = {
                    "checkpoint": step_number,
                    "loss": log_entry.get("Loss since last update"),
                    "step_timestamp": log_entry.get("timestamp"),
                }

            # Look for validation entries with evaluation metrics
            elif "Validation set statistics" in message:
                # Extract all keys except the basic ones (like "message" and "timestamp")
                validation_metrics = {
                    key: value for key, value in log_entry.items() if key not in {"message", "timestamp"}
                }
                # Optionally, you might store the timestamp of the validation entry separately.
                validation_metrics["validation_timestamp"] = log_entry.get("timestamp")

                # Merge the checkpoint info (if any) with the validation metrics.
                # If no checkpoint was found before, you could either skip or record a row with NaNs.
                if current_checkpoint:
                    # Combine the dictionaries; if the same key exists, validation metrics overwrite.
                    combined_entry = {
                        **current_checkpoint,
                        **validation_metrics,
                    }
                    results.append(combined_entry)
                    # Clear current_checkpoint so that a new pair is started for the next checkpoint.
                    current_checkpoint = {}
                else:
                    # In case a validation entry occurs without a preceding checkpoint entry,
                    # record it with a missing checkpoint.
                    row = {
                        "checkpoint": None,
                        "loss": None,
                        "step_timestamp": None,
                    }
                    row.update(validation_metrics)
                    results.append(row)

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"Reading {log_file = } DONE",  # noqa: G004 - low overhead
        )

    # Convert the list of dictionaries into a pandas DataFrame.
    parsed_df = pd.DataFrame(
        data=results,
    )

    if verbosity >= Verbosity.NORMAL:
        log_dataframe_info(
            df=parsed_df,
            df_name="parsed_df",
            logger=logger,
        )

    return parsed_df
