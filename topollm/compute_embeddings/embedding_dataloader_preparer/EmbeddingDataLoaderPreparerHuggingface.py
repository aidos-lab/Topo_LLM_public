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

from functools import partial

import datasets
import torch.utils.data

from topollm.compute_embeddings.embedding_dataloader_preparer.EmbeddingDataLoaderPreparerABC import (
    EmbeddingDataLoaderPreparer,
)
from topollm.logging.log_dataset_info import log_huggingface_dataset_info


class EmbeddingDataLoaderPreparerHuggingface(EmbeddingDataLoaderPreparer):
    @property
    def sequence_length(
        self,
    ) -> int:
        return self.preparer_context.tokenizer_config.max_length

    def __len__(
        self,
    ) -> int:
        """Returns the number of samples in the dataset."""

        return len(self.dataset_preparer)

    def create_dataset_tokenized(
        self,
        dataset: datasets.Dataset,
    ) -> datasets.Dataset:
        """Tokenizes dataset."""
        # Make a partial function for mapping tokenizer over the dataset
        partial_map_fn = partial(
            self.convert_dataset_entry_to_features,
            tokenizer=self.preparer_context.tokenizer,
            column_name=self.preparer_context.data_config.column_name,
            max_length=self.sequence_length,
        )

        dataset_tokenized = dataset.map(
            partial_map_fn,
            batched=True,
            batch_size=self.preparer_context.embeddings_config.dataset_map.batch_size,
            num_proc=self.preparer_context.embeddings_config.dataset_map.num_proc,
        )

        if self.verbosity >= 1:
            self.logger.info(
                f"{dataset_tokenized = }",
            )
            log_huggingface_dataset_info(
                dataset=dataset_tokenized,
                dataset_name="dataset_tokenized",
                logger=self.logger,
            )

        return dataset_tokenized

    def create_dataloader_from_tokenized_dataset(
        self,
        dataset_tokenized: datasets.Dataset,
    ) -> torch.utils.data.DataLoader:
        # The mapped dataset has the input_ids and attention_mask
        # as lists of integers, but we want to convert them to torch tensors
        # to use them as model input.
        # We will take care of this in the collate function of the DataLoader,
        # which will also move the data to the appropriate device.
        #
        # An alternative way to set the format of the dataset to torch tensors
        # is given below:
        #
        # dataset_tokenized.set_format(
        #     type="torch",
        #     columns=[
        #         "input_ids",
        #         "attention_mask",
        #     ],
        # )

        # The multiprocessing_context argument is the solution taken from:
        # https://github.com/pytorch/pytorch/issues/87688
        # But it does not appear to work with the "mps" backend.
        #
        # Not that you need to set `num_workers=0` so that the data loading
        # runs in the main process.
        # This appears to be necessary with the "mps" backend.
        dataloader = torch.utils.data.DataLoader(
            dataset_tokenized,  # type: ignore
            batch_size=self.preparer_context.embeddings_config.batch_size,
            shuffle=False,
            collate_fn=self.preparer_context.collate_fn,
            num_workers=self.preparer_context.embeddings_config.num_workers,
            # multiprocessing_context=(
            #     "fork" if torch.backends.mps.is_available() else None
            # ),
        )

        if self.verbosity >= 1:
            self.logger.info(
                f"{dataloader = }",
            )

        return dataloader

    def prepare_dataloader(
        self,
    ) -> torch.utils.data.DataLoader:
        """
        Loads and prepares a dataset,
        returns a dataloader.
        """

        dataset = self.dataset_preparer.prepare_dataset()

        dataset_tokenized = self.create_dataset_tokenized(
            dataset=dataset,
        )

        dataloader = self.create_dataloader_from_tokenized_dataset(
            dataset_tokenized=dataset_tokenized,
        )

        return dataloader
