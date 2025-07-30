"""Run inference with a language model."""

import logging
import pprint
from typing import TYPE_CHECKING

import torch
import transformers
from transformers.modeling_utils import PreTrainedModel

from topollm.config_classes.main_config import MainConfig
from topollm.model_handling.prepare_loaded_model_container import (
    prepare_device_and_tokenizer_and_model_from_main_config,
)
from topollm.model_inference.causal_language_modeling.do_text_generation import (
    do_text_generation,
)
from topollm.model_inference.default_prompts import (
    get_default_clm_prompts,
    get_default_mlm_prompts,
)
from topollm.model_inference.masked_language_modeling.do_fill_mask import do_fill_mask
from topollm.typing.enums import LMmode, Verbosity

if TYPE_CHECKING:
    from topollm.model_handling.loaded_model_container import LoadedModelContainer

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def do_inference(
    main_config: MainConfig,
    prompts: list[str] | None = None,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> list[list]:
    """Run inference with a language model.

    If `prompts` is `None`, default prompts are used.
    Make sure to not accidentally use an empty list as the default argument.
    """
    loaded_model_container: LoadedModelContainer = prepare_device_and_tokenizer_and_model_from_main_config(
        main_config=main_config,
        verbosity=verbosity,
        logger=logger,
    )
    device: torch.device = loaded_model_container.device
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast = (
        loaded_model_container.tokenizer
    )
    lm_mode: LMmode = loaded_model_container.lm_mode
    model: PreTrainedModel = loaded_model_container.model

    # Set up the model for evaluation.
    model.eval()

    if lm_mode == LMmode.MLM:
        if prompts is None:
            prompts = get_default_mlm_prompts(
                mask_token=tokenizer.mask_token,  # type: ignore - problem with inferring the correct type of the mask token
            )
        logger.info(
            "prompts:\n%s",
            pprint.pformat(prompts),
        )

        results = do_fill_mask(
            tokenizer=tokenizer,
            model=model,
            prompts=prompts,
            device=device,
            logger=logger,
        )
    elif lm_mode == LMmode.CLM:
        if prompts is None:
            prompts = get_default_clm_prompts()
        logger.info(
            "prompts:\n%s",
            pprint.pformat(prompts),
        )

        results = do_text_generation(
            tokenizer=tokenizer,
            model=model,
            prompts=prompts,
            max_length=main_config.inference.max_length,
            num_return_sequences=main_config.inference.num_return_sequences,
            device=device,
            logger=logger,
        )
    else:
        msg: str = f"Invalid lm_mode: {lm_mode = }"
        raise ValueError(msg)

    # TODO: Assemble prompts and results into a dictionary; save the result to disk in json format

    return results
