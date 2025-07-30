"""Fill mask in a masked language model."""

import logging
import pprint

import torch
import transformers

default_device = torch.device(
    device="cpu",
)
logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def do_fill_mask(
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    model: transformers.PreTrainedModel,
    prompts: list[str],
    device: torch.device = default_device,
    logger: logging.Logger = logger,
) -> list[list[dict]]:
    """Fill mask in a masked language model."""
    fill_pipeline: transformers.Pipeline = transformers.pipeline(
        task="fill-mask",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    logger.info(
        msg=f"prompts:\n{pprint.pformat(prompts)}",  # noqa: G004 - low overhead
    )

    result = fill_pipeline(
        inputs=prompts,
    )
    logger.info(
        msg=f"result:\n{pprint.pformat(result)}",  # noqa: G004 - low overhead
    )

    return result  # type: ignore - problem with matching return type of pipeline
