# Copyright 2024-2025
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


"""Factory for creating a dataset filter."""

import logging

from topollm.config_classes.data.data_config import DataConfig
from topollm.data_handling.dataset_filtering.dataset_filter_basic import DatasetFilterBasic
from topollm.data_handling.dataset_filtering.protocol import DatasetFilter
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_dataset_filter(
    data_config: DataConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> DatasetFilter:
    """Get a dataset filter.

    Note that we acre using the data_config instead of the more specific filter config,
    since to configure to filter, we need to know the column_name which determines which data is embedded.
    """
    result = DatasetFilterBasic(
        data_filtering_config=data_config.filtering,
        column_name=data_config.column_name,
        verbosity=verbosity,
        logger=logger,
    )

    return result
