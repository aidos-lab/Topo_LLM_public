"""Generate text based on the provided prompts using causal language modeling."""

import logging
import pprint

import torch
import transformers
from tqdm import tqdm

default_device = torch.device(device="cpu")
default_logger: logging.Logger = logging.getLogger(name=__name__)


def do_text_generation(
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    model: transformers.PreTrainedModel,
    prompts: list[str],
    max_length: int = 50,
    num_return_sequences: int = 3,
    device: torch.device = default_device,
    logger: logging.Logger = default_logger,
) -> list[list[str]]:
    """Generate text based on the provided prompts using causal language modeling.

    Support for multiple generations per prompt.

    Args:
    ----
        tokenizer: A tokenizer instance compatible with the provided model.
        model: A pre-trained model instance capable of text generation.
        prompts: A list of strings, each being a prompt to generate text from.
        max_length: Maximum length of the generated text.
        num_return_sequences: Number of sequences to generate for each prompt.
        device: The device to run the model on.
        logger: Logger instance for logging information.

    Returns:
    -------
        A list of lists containing generated text sequences for each prompt.

    """
    # Notes:
    # - We set `do_sample=True` to avoid problem with the `num_return_sequences` parameter,
    # see: https://github.com/huggingface/transformers/issues/2415
    text_generation_pipeline: transformers.Pipeline = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
    )

    logger.info(
        "prompts:\n%s",
        pprint.pformat(prompts),
    )

    all_generated_texts: list[list[str]] = []

    for prompt in tqdm(
        prompts,
        desc="Iterating over prompts",
    ):
        results: list[dict] = text_generation_pipeline(
            prompt,
        )  # type: ignore - problem with the type hint in the transformers library

        if results is None:
            msg = "No results were generated."
            raise TypeError(msg)
        if not isinstance(
            results,
            list,
        ):
            msg: str = f"{results = } must be a list."
            raise TypeError(msg)

        generated_texts: list[str] = [result["generated_text"] for result in results]

        # Appending the generated texts for the current prompt to the list of lists
        all_generated_texts.append(
            generated_texts,
        )

        # Logging each generated text for the current prompt
        logger.info(
            msg=f"{prompt = }",  # noqa: G004 - low overhead
        )
        logger.info(
            "generated_texts:\n%s",
            pprint.pformat(generated_texts),
        )

    return all_generated_texts
