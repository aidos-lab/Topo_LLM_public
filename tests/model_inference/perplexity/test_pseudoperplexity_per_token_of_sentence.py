import logging

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from topollm.config_classes.tokenizer.tokenizer_config import TokenizerConfig
from topollm.model_inference.perplexity.compute_perplexity_over_dataset import pseudoperplexity_per_token_of_sentence
from topollm.typing.enums import (
    MLMPseudoperplexityGranularity,
    Verbosity,
)


def test_pseudoperplexity_per_token_of_sentence(
    tokenizer_config: TokenizerConfig,
    device_fixture: torch.device,
    verbosity: Verbosity,
    logger_fixture: logging.Logger,
) -> None:
    """Test the pseudoperplexity_per_token_of_sentence function."""
    model_name = "roberta-base"
    model = AutoModelForMaskedLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
    )

    model.to(
        device_fixture,
    )
    model.eval()

    sentences_list = [
        "Paris is in France.",
        "Berlin is in France",
        "London is the capital of Great Britain.",
        "London is the capital of South America.",
    ]

    for sentence in sentences_list:
        result = pseudoperplexity_per_token_of_sentence(
            sentence=sentence,
            tokenizer=tokenizer,
            tokenizer_config=tokenizer_config,
            model=model,
            mlm_pseudoperplexity_granularity=MLMPseudoperplexityGranularity.TOKEN,
            device=device_fixture,
            verbosity=verbosity,
            logger=logger_fixture,
        )
        logger_fixture.info(
            "sentence:\n%s",
            sentence,
        )
        logger_fixture.info(
            "result:\n%s",
            result,
        )
