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

"""Basic dataset filter."""

import logging
import pprint

import datasets

from topollm.config_classes.data.data_filtering_config import DataFilteringConfig
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class DatasetFilterBasic:
    """Basic dataset filter."""

    def __init__(
        self,
        data_filtering_config: DataFilteringConfig,
        column_name: str,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the dataset filter.

        If `remove_empty_sequences` is True,
        dataset entries with empty sequences in the column `column_name` will be removed.
        """
        self.data_filtering_config: DataFilteringConfig = data_filtering_config
        self.column_name: str = column_name

        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"Using {self.__class__.__name__} as dataset filter.",  # noqa: G004 - low overhead
            )
            logger.info(
                msg=f"data_filtering_config:\n{pprint.pformat(object=data_filtering_config)}",  # noqa: G004 - low overhead
            )
            logger.info(
                msg=f"Filtering based on data in {column_name = }",  # noqa: G004 - low overhead
            )

    def filter_dataset_dict(
        self,
        dataset_dict: datasets.DatasetDict,
    ) -> datasets.DatasetDict:
        """Filter a dataset dict."""
        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                "dataset_dict:\n%s",
                dataset_dict,
            )

        # # # #
        # Potentially remove empty sequences,
        # where the sequence is the column `column_name`.
        # We expect the dataset_dict to have a column with the name `column_name` which has a length function.
        if self.data_filtering_config.remove_empty_sequences:
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg=f"Removing empty sequences based on {self.column_name = }.",  # noqa: G004 - low overhead
                )

            dataset_dict_filtered: datasets.DatasetDict = dataset_dict.filter(
                function=lambda x: len(x[self.column_name]) > 0,
            )
        else:
            if self.verbosity >= Verbosity.NORMAL:
                self.logger.info(
                    msg="Not removing empty sequences.",
                )
            dataset_dict_filtered = dataset_dict

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="Logging information of dataset_dict_filtered after potentially removing empty sequences",
            )
            self.logger.info(
                "dataset_dict_filtered:\n%s",
                dataset_dict_filtered,
            )

        # # # #
        # Potentially remove sequences with starting segment in the blocklist.
        # If the blocklist is empty, no sequences will be removed.
        blocklist: list[str] = self.data_filtering_config.remove_sequences_with_starting_segment_in_this_blocklist
        if blocklist:
            # We expect the dataset_dict to have a column with the name `column_name` which has a startswith function.
            for blocklist_item in blocklist:
                if self.verbosity >= Verbosity.NORMAL:
                    self.logger.info(
                        msg=f"Removing sequences with starting segment in blocklist item {blocklist_item = }.",  # noqa: G004 - low overhead
                    )

                def current_filter_function(
                    x: dict,
                    blocklist_item: str = blocklist_item,  # Use default argument to avoid late binding
                ) -> bool:
                    return not x[self.column_name].startswith(blocklist_item)

                dataset_dict_filtered = dataset_dict_filtered.filter(
                    function=current_filter_function,
                )

                if self.verbosity >= Verbosity.NORMAL:
                    self.logger.info(
                        msg=f"Logging information of dataset_dict_filtered after removing sequences "  # noqa: G004 - low overhead
                        f"with starting segment in blocklist item {blocklist_item = }.",
                    )
                    self.logger.info(
                        "dataset_dict_filtered:\n%s",
                        dataset_dict_filtered,
                    )

        if self.verbosity >= Verbosity.NORMAL:
            self.logger.info(
                msg="Logging information of dataset_dict_filtered at the end of filtering.",
            )
            self.logger.info(
                "dataset_dict_filtered:\n%s",
                dataset_dict_filtered,
            )

        return dataset_dict_filtered
