"""Get the token ids to filter out from the tokenizer based on the filter tokens config."""

import logging

from topollm.config_classes.embeddings_data_prep.filter_tokens_config import FilterTokensConfig
from topollm.typing.enums import Verbosity
from topollm.typing.types import TransformersTokenizer

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def get_token_ids_from_filter_tokens_config(
    tokenizer: TransformersTokenizer,
    filter_tokens_config: FilterTokensConfig,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> list[int]:
    """Get the token ids to filter out from the tokenizer based on the filter tokens config."""
    token_ids_to_filter: list = []
    if filter_tokens_config.remove_bos_token:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{tokenizer.bos_token_id = }",  # noqa: G004 - low overhead
            )
            logger.info(
                msg=f"{tokenizer.bos_token = }",  # noqa: G004 - low overhead
            )
        if tokenizer.bos_token_id is None:
            logger.warning(
                msg="The config specifies that the bos_token should be filtered, "
                "but the tokenizer beginning of sequence token id is None. "
                "The script will continue here, but we will NOT filter the bos_token.",
            )
        else:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"The tokenizer beginning of sequence token id {tokenizer.bos_token_id = } will be filtered.",  # noqa: G004 - low overhead
                )
            token_ids_to_filter.append(
                tokenizer.bos_token_id,
            )
    if filter_tokens_config.remove_eos_token:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{tokenizer.eos_token_id = }",  # noqa: G004 - low overhead
            )
            logger.info(
                msg=f"{tokenizer.eos_token = }",  # noqa: G004 - low overhead
            )
        if tokenizer.eos_token_id is None:
            logger.warning(
                msg="The config specifies that the eos_token should be filtered, "
                "but the tokenizer end of sequence token id is None. "
                "The script will continue here, but will NOT filter the eos_token.",
            )
        else:
            if verbosity >= Verbosity.NORMAL:
                logger.info(
                    msg=f"The tokenizer end of sequence token id {tokenizer.eos_token_id = } will be filtered.",  # noqa: G004 - low overhead
                )
            token_ids_to_filter.append(
                tokenizer.eos_token_id,
            )
    if filter_tokens_config.remove_pad_token:
        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"{tokenizer.pad_token_id = }",  # noqa: G004 - low overhead
            )
            logger.info(
                msg=f"{tokenizer.pad_token = }",  # noqa: G004 - low overhead
            )
        if tokenizer.pad_token_id is None:
            msg = "The tokenizer padding token id is None.Since this is probably not intended, we will raise an error."
            raise ValueError(
                msg,
            )

        if verbosity >= Verbosity.NORMAL:
            logger.info(
                msg=f"The tokenizer padding token id {tokenizer.pad_token_id = } will be filtered.",  # noqa: G004 - low overhead
            )
        token_ids_to_filter.append(
            tokenizer.pad_token_id,
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            msg=f"{token_ids_to_filter = }",  # noqa: G004 - low overhead
        )

    return token_ids_to_filter
