# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (mail@ruppik.net)
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

"""Extract the losses of a model over the finetuning checkpoints."""

import pathlib

import pandas as pd

from topollm.config_classes.constants import NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS


class ModelLossExtractor:
    """Class for extracting the losses of a model over the finetuning checkpoints."""

    def __init__(
        self,
        train_loss_file_path: pathlib.Path,
        eval_loss_file_path: pathlib.Path,
    ) -> None:
        """Initialize the class."""
        # Check that the files exist
        if not train_loss_file_path.exists():
            msg = f"The file {train_loss_file_path = } does not exist."
            raise FileNotFoundError(msg)
        if not eval_loss_file_path.exists():
            msg = f"The file {eval_loss_file_path = } does not exist."
            raise FileNotFoundError(msg)

        train_loss_df: pd.DataFrame = pd.read_csv(
            filepath_or_buffer=train_loss_file_path,
        )
        eval_loss_df: pd.DataFrame = pd.read_csv(
            filepath_or_buffer=eval_loss_file_path,
        )
        self.loss_dfs_container: dict = {
            "train": train_loss_df,
            "eval": eval_loss_df,
        }

    def get_model_losses_over_finetuning_checkpoints(
        self,
        data_full: str,
        data_subsampling_split: str,
        model_partial_name: str,
        language_model_seed: int,
    ) -> pd.DataFrame | None:
        """Extract the losses of a model over the finetuning checkpoints."""
        match data_subsampling_split:
            case "train":
                selected_table = self.loss_dfs_container["train"]
                column_name_split_descriptor = "train"
            case "validation":
                selected_table = self.loss_dfs_container["eval"]
                column_name_split_descriptor = "eval"
            case _:
                return None

        match model_partial_name:
            case "model=model-roberta-base_task-masked_lm_multiwoz21-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50":
                if data_full not in [
                    "data=multiwoz21_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags",
                    "data=multiwoz21_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags",
                ]:
                    # We currently only have losses for this data_full for this model.
                    # The two different data_full values are due to the fact that we changed the naming scheme,
                    # but the data is the same.
                    return None
                loss_column_name = f"fresh-microwave-2 - {column_name_split_descriptor}/loss"
            case "model=model-roberta-base_task-masked_lm_one-year-of-tsla-on-reddit-train-10000-ner_tags_ftm-standard_lora-None_5e-05-constant-0.01-50":
                if data_full not in [
                    "data=one-year-of-tsla-on-reddit_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags",
                    "data=one-year-of-tsla-on-reddit_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags",
                ]:
                    # We currently only have losses for this data_full for this model
                    # The two different data_full values are due to the fact that we changed the naming scheme,
                    # but the data is the same.
                    return None
                loss_column_name = f"dutiful-snowball-3 - {column_name_split_descriptor}/loss"
            case _:
                return None

        if language_model_seed != 1234:
            # We currently only have losses for the seed 1234
            return None

        # Create the output dataframe from the selected table and the loss_column_name
        output_table = selected_table[["train/global_step", loss_column_name]]

        output_table = output_table.rename(
            columns={
                "train/global_step": NAME_PREFIXES_TO_FULL_AUGMENTED_DESCRIPTIONS["ckpt"],
                loss_column_name: "loss",
            },
        )

        return output_table
