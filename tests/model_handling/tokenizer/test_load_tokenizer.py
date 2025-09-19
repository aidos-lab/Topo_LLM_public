import logging
import os

import pytest
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from tests.model_handling.parameter_lists import (
    example_pretrained_model_name_or_path_list,
)
from topollm.config_classes.tokenizer.tokenizer_config import TokenizerConfig
from topollm.model_handling.tokenizer.load_tokenizer import load_tokenizer
from topollm.typing.enums import Verbosity


@pytest.mark.parametrize(
    "pretrained_model_name_or_path",
    example_pretrained_model_name_or_path_list,
)
@pytest.mark.uses_transformers_models()
def test_load_tokenizer(
    pretrained_model_name_or_path: str | os.PathLike,
    tokenizer_config: TokenizerConfig,
    verbosity: Verbosity,
    logger_fixture: logging.Logger,
) -> None:
    tokenizer = load_tokenizer(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        tokenizer_config=tokenizer_config,
        verbosity=verbosity,
        logger=logger_fixture,
    )

    assert tokenizer is not None  # noqa: S101 - pytest assert
    assert isinstance(  # noqa: S101 - pytest assert
        tokenizer,
        PreTrainedTokenizer | PreTrainedTokenizerFast,
    )
