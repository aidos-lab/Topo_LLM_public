"""Factory for TokenMasker."""

import logging

from topollm.config_classes.embeddings_data_prep.token_masking_config import TokenMaskingConfig
from topollm.embeddings_data_prep.token_masker.protocol import TokenMasker
from topollm.embeddings_data_prep.token_masker.token_masker_no_masking import TokenMaskerNoMasking
from topollm.typing.enums import EmbeddingsDataPrepTokenMaskingMode, Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_token_masker(
    token_masking_config: TokenMaskingConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> TokenMasker:
    """Get a TokenMasker instance."""
    match token_masking_config.token_masking_mode:
        case EmbeddingsDataPrepTokenMaskingMode.NO_MASKING:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg="Using no masking via TokenMaskerNoMasking.",
                )
            result = TokenMaskerNoMasking(
                token_masking_config=token_masking_config,
                verbosity=verbosity,
                logger=logger,
            )
        case EmbeddingsDataPrepTokenMaskingMode.BASED_ON_META_COLUMN:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg="Using masking based on meta column via TokenMaskerBasedOnMetaColumn.",
                )

            # TODO: Implement TokenMaskerBasedOnMetaColumn and include in factory

            msg = "TokenMaskerBasedOnMetaColumn not implemented yet."
            raise NotImplementedError(msg)
        case _:
            msg: str = f"Token masking mode {token_masking_config.token_masking_mode = } not supported."
            raise ValueError(
                msg,
            )

    return result
