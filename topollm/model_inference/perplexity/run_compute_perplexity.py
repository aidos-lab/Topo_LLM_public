# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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

"""Compute the (pseudo-)perplexity of a masked language model and save to disk."""

import logging
from typing import TYPE_CHECKING

import hydra
import hydra.core.hydra_config
import numpy as np
import omegaconf
import torch
import transformers
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from topollm.config_classes.setup_OmegaConf import setup_OmegaConf
from topollm.logging.initialize_configuration_and_log import initialize_configuration
from topollm.logging.setup_exception_logging import setup_exception_logging
from topollm.model_handling.loaded_model_container import LoadedModelContainer
from topollm.model_handling.prepare_device_and_tokenizer_and_model import prepare_device_and_tokenizer_and_model
from topollm.typing.enums import LMmode

if TYPE_CHECKING:
    from topollm.config_classes.main_config import MainConfig

default_device = torch.device("cpu")
default_logger = logging.getLogger(__name__)

global_logger = logging.getLogger(__name__)

setup_exception_logging(
    logger=global_logger,
)


setup_OmegaConf()


def pseudoperplexity_per_token_of_sentence(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    model: PreTrainedModel,
    sentence: str,
    device: torch.device = default_device,
) -> torch.Tensor:
    """Compute the pseudo-perplexity of a masked language model on a given sentence."""
    mask_token_id = tokenizer.mask_token_id
    if not isinstance(
        mask_token_id,
        int,
    ):
        msg = "Expected an integer."
        raise TypeError(msg)

    tensor_input = tokenizer.encode(
        sentence,
        return_tensors="pt",
    )
    # Example:
    # `model = 'roberta-base'`
    # `sentence = 'Paris is in France.'
    # `tensor_input = tensor([[    0, 32826,    16,    11,  1470,     4,     2]])`
    # `[tokenizer.decode(id) for id in tensor_input[0]] = ['<s>', 'Paris', ' is', ' in', ' France', '.', '</s>']`

    if not isinstance(
        tensor_input,
        torch.Tensor,
    ):
        msg = "Expected a torch.Tensor."
        raise TypeError(msg)

    repeat_input = tensor_input.repeat(
        tensor_input.size(-1) - 2,
        1,
    )
    # repeat_input =
    # tensor([[    0, 32826,    16,    11,  1470,     4,     2],
    #         [    0, 32826,    16,    11,  1470,     4,     2],
    #         [    0, 32826,    16,    11,  1470,     4,     2],
    #         [    0, 32826,    16,    11,  1470,     4,     2],
    #         [    0, 32826,    16,    11,  1470,     4,     2]])

    diagonal_mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    # diagonal_mask =
    # tensor([[0., 1., 0., 0., 0., 0., 0.],
    #         [0., 0., 1., 0., 0., 0., 0.],
    #         [0., 0., 0., 1., 0., 0., 0.],
    #         [0., 0., 0., 0., 1., 0., 0.],
    #         [0., 0., 0., 0., 0., 1., 0.]])

    masked_input = repeat_input.masked_fill(
        mask=(diagonal_mask == 1),
        value=mask_token_id,
    )
    # masked_input =
    # tensor([[    0, 50264,    16,    11,  1470,     4,     2],
    #         [    0, 32826, 50264,    11,  1470,     4,     2],
    #         [    0, 32826,    16, 50264,  1470,     4,     2],
    #         [    0, 32826,    16,    11, 50264,     4,     2],
    #         [    0, 32826,    16,    11,  1470, 50264,     2]])

    labels = repeat_input.masked_fill(
        mask=(masked_input != mask_token_id),
        value=-100,
    )
    # labels =
    # tensor([[ -100, 32826,  -100,  -100,  -100,  -100,  -100],
    #         [ -100,  -100,    16,  -100,  -100,  -100,  -100],
    #         [ -100,  -100,  -100,    11,  -100,  -100,  -100],
    #         [ -100,  -100,  -100,  -100,  1470,  -100,  -100],
    #         [ -100,  -100,  -100,  -100,  -100,     4,  -100]])

    # Move inputs and labels to the correct device.
    masked_input = masked_input.to(device)
    labels = labels.to(device)

    with torch.inference_mode():
        # TODO(Ben): Move model, input and labels to the correct device.
        output = model(
            masked_input,
            labels=labels,
        )

        # TODO(Ben): To obtain the token-level loss/perplexity, we might need to input thes sequences separately
        loss = output.loss

    return loss


def token_level_to_sentence_level_pseudoperplexity(
    loss: torch.Tensor,
):
    return np.exp(loss.item())


def compute_perplexity_over_dataset(
    device: torch.device,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    lm_mode: LMmode,
    model: transformers.PreTrainedModel,
):
    # TODO(Ben): Load the correct dataset
    # TODO(Ben): Save the token-level (pseudo-)perplexity to an array.

    if lm_mode == LMmode.CLM:
        msg = "Perplexity computation not implemented for CLM yet."
        raise NotImplementedError(msg)

    # # # #
    result = pseudoperplexity_per_token_of_sentence(
        tokenizer=tokenizer,
        model=model,
        device=device,
        sentence="Paris is in France.",
        # sentence="London is the capital of Great Britain.",
    )

    print(result)
    # 4.541251105675365

    # # # #
    result = pseudoperplexity_per_token_of_sentence(
        tokenizer=tokenizer,
        model=model,
        device=device,
        sentence="London is the capital of South America.",
    )
    print(result)
    # 6.162017238332462


@hydra.main(
    config_path="../../../configs",
    config_name="main_config",
    version_base="1.2",
)
def main(
    config: omegaconf.DictConfig,
) -> None:
    """Run the script."""
    logger = global_logger
    logger.info("Running script ...")

    main_config: MainConfig = initialize_configuration(
        config=config,
        logger=logger,
    )

    loaded_model_container: LoadedModelContainer = prepare_device_and_tokenizer_and_model(
        main_config=main_config,
        logger=logger,
    )
    model = loaded_model_container.model
    # Put model in evaluation mode
    model.eval()

    compute_perplexity_over_dataset(
        device=loaded_model_container.device,
        tokenizer=loaded_model_container.tokenizer,
        lm_mode=loaded_model_container.lm_mode,
        model=model,
    )

    logger.info("Running script DONE")


if __name__ == "__main__":
    main()
