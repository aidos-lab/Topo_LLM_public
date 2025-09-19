"""Factory function to get the local estimates filter."""

import logging

from topollm.analysis.local_estimates_handling.deduplicator.array_deduplicator import ArrayDeduplicator
from topollm.analysis.local_estimates_handling.deduplicator.identity_deduplicator import IdentityDeduplicator
from topollm.analysis.local_estimates_handling.deduplicator.protocol import PreparedDataDeduplicator
from topollm.config_classes.local_estimates.filtering_config import LocalEstimatesFilteringConfig
from topollm.typing.enums import DeduplicationMode, Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_prepared_data_deduplicator(
    local_estimates_filtering_config: LocalEstimatesFilteringConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> PreparedDataDeduplicator:
    """Get the filter for the local estimates computation."""
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Getting prepared data deduplicator ...",
        )

    match local_estimates_filtering_config.deduplication_mode:
        case DeduplicationMode.IDENTITY:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg="Returning IdentityDeduplicator ...",
                )
            deduplicator = IdentityDeduplicator()
        case DeduplicationMode.ARRAY_DEDUPLICATOR:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg="Returning ArrayDeduplicator ...",
                )
            deduplicator = ArrayDeduplicator(
                verbosity=verbosity,
                logger=logger,
            )
        case _:
            msg: str = f"Unknown {local_estimates_filtering_config.deduplication_mode = }"
            raise ValueError(
                msg,
            )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg="Getting prepared data deduplicator DONE",
        )

    return deduplicator
