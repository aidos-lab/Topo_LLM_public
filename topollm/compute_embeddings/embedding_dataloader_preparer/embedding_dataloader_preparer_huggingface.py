"""Prepare a dataloader for computing embeddings using Huggingface datasets."""

from functools import partial

import datasets
import torch.utils.data

from topollm.compute_embeddings.embedding_dataloader_preparer.protocol import (
    EmbeddingDataLoaderPreparer,
)
from topollm.logging.log_dataset_info import log_huggingface_dataset_info
from topollm.typing.enums import Verbosity


class EmbeddingDataLoaderPreparerHuggingfaceWithTokenization(EmbeddingDataLoaderPreparer):
    """Prepare a dataloader for computing embeddings using Huggingface datasets."""

    @property
    def sequence_length(
        self,
    ) -> int:
        """Return the sequence length for the model."""
        # Note:
        # For the pre-tokenized datasets, since the padding in the collate function
        # might change the sequence length, we cannot infer the sequence length
        # from the dataset at this point.

        # Get the sequence length from the tokenizer config
        result: int = self.preparer_context.tokenizer_config.max_length

        return result

    def __len__(
        self,
    ) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset_preparer)

    def create_dataset_tokenized(
        self,
        dataset: datasets.Dataset,
    ) -> datasets.Dataset:
        """Tokenizes dataset."""
        # Make a partial function for mapping tokenizer over the dataset.
        # Note that the max_length parameter is taken from the config,
        # but depending on the convert_dataset_entry_to_features_function,
        # it might not be used.
        # We cannot use self.sequence_length here, because this might lead to infinite recursion.
        partial_map_fn = partial(
            self.convert_dataset_entry_to_features_function,
            tokenizer=self.preparer_context.tokenizer,
            column_name=self.preparer_context.data_config.column_name,
            max_length=self.preparer_context.tokenizer_config.max_length,
        )

        dataset_tokenized: datasets.Dataset = dataset.map(
            function=partial_map_fn,
            batched=True,
            batch_size=self.preparer_context.embeddings_config.dataset_map.batch_size,
            num_proc=self.preparer_context.embeddings_config.dataset_map.num_proc,
            keep_in_memory=True,  # This avoids caching the dataset on disk
        )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                "dataset_tokenized:\n%s",
                dataset_tokenized,
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
        """Create a dataloader from a tokenized dataset.

        The mapped dataset has the input_ids and attention_mask
        as lists of integers, but we want to convert them to torch tensors
        to use them as model input.
        We will take care of this in the collate function of the DataLoader,
        which will also move the data to the appropriate device.

        An alternative way to set the format of the dataset to torch tensors
        is given below:

        dataset_tokenized.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
            ],
        )
        """
        # The multiprocessing_context argument is the solution taken from:
        # https://github.com/pytorch/pytorch/issues/87688
        # But it does not appear to work with the "mps" backend.
        # > multiprocessing_context=(
        # >     "fork" if torch.backends.mps.is_available() else None
        # > ),
        #
        # Not that you need to set `num_workers=0` so that the data loading
        # runs in the main process.
        # This appears to be necessary with the "mps" backend.
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset_tokenized,  # type: ignore - typing issue with Dataset
            batch_size=self.preparer_context.embeddings_config.batch_size,
            shuffle=False,
            collate_fn=self.preparer_context.collate_fn,
            num_workers=self.preparer_context.embeddings_config.num_workers,
        )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                "dataloader:\n%s",
                dataloader,
            )

        return dataloader

    def get_dataset(
        self,
    ) -> datasets.Dataset:
        """Prepare dataset if not already prepared, otherwise return it."""
        if self._dataset is None:
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg="Preparing dataset ...",
                )
            self._dataset = self.dataset_preparer.prepare_dataset()
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg="Preparing dataset DONE",
                )
        elif self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="Dataset already prepared. Returning it.",
            )

        return self._dataset

    def get_dataset_tokenized(
        self,
    ) -> datasets.Dataset:
        """Prepare tokenized dataset if not already prepared, otherwise return it."""
        if self._dataset_tokenized is None:
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg="Preparing tokenized dataset ...",
                )
            self._dataset_tokenized = self.create_dataset_tokenized(
                dataset=self.get_dataset(),
            )
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg="Preparing tokenized dataset DONE",
                )
        elif self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="Tokenized dataset already prepared. Returning it.",
            )

        return self._dataset_tokenized

    def get_dataloader(
        self,
    ) -> torch.utils.data.DataLoader:
        """Prepare dataloader if not already prepared, otherwise return it."""
        if self._dataloader is None:
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg="Preparing dataloader ...",
                )
            self._dataloader = self.create_dataloader_from_tokenized_dataset(
                dataset_tokenized=self.get_dataset_tokenized(),
            )
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg="Preparing dataloader DONE",
                )
        elif self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="Dataloader already prepared. Returning it.",
            )

        return self._dataloader
