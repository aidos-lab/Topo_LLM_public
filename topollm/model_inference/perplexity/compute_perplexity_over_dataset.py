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

"""Iterate over a dataset and compute the perplexity for each sentence."""

import logging

import datasets
import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from topollm.config_classes.tokenizer.tokenizer_config import TokenizerConfig
from topollm.model_handling.loaded_model_container import LoadedModelContainer
from topollm.model_inference.perplexity.sentence_perplexity_container import SentencePerplexityContainer
from topollm.typing.enums import LMmode, MLMPseudoperplexityGranularity, Verbosity
from topollm.typing.types import PerplexityResultsList

default_device = torch.device("cpu")
default_logger = logging.getLogger(__name__)


def pseudoperplexity_per_token_of_sentence(
    sentence: str,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    tokenizer_config: TokenizerConfig,
    model: PreTrainedModel,
    mlm_pseudoperplexity_mode: MLMPseudoperplexityGranularity = MLMPseudoperplexityGranularity.SENTENCE,
    device: torch.device = default_device,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> SentencePerplexityContainer:
    """Compute the pseudo-perplexity of a masked language model on a given sentence."""
    mask_token_id = tokenizer.mask_token_id
    if not isinstance(
        mask_token_id,
        int,
    ):
        msg = "Expected an integer."
        raise TypeError(msg)

    # Make sure that `padding=False`, otherwise the repeated input will be duplicated many times.
    tensor_input = tokenizer.encode(
        sentence,
        return_tensors="pt",
        max_length=tokenizer_config.max_length,
        padding=False,
        truncation="longest_first",
    )
    # Example:
    # `model = 'roberta-base'`
    # `sentence = 'Paris is in France.'
    # `tensor_input = tensor([[    0, 32826,    16,    11,  1470,     4,     2]])`
    # `[tokenizer.decode(single_token_id) for single_token_id in tensor_input[0]] = ['<s>', 'Paris', ' is', ' in', ' France', '.', '</s>']`

    if not isinstance(
        tensor_input,
        torch.Tensor,
    ):
        msg = "Expected a torch.Tensor."
        raise TypeError(msg)

    token_id_list = tensor_input[0].tolist()  # type: ignore - tensor_input can be subscripted
    tensor_input_decoded = [tokenizer.decode(single_token_id) for single_token_id in tensor_input[0]]  # type: ignore - tensor_input can be subscripted

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

    results_loss_list: list[float] = []

    if mlm_pseudoperplexity_mode == MLMPseudoperplexityGranularity.SENTENCE:
        # We send the entire batch at once through the model to get a sentence-level loss.
        with torch.inference_mode():
            output = model(
                masked_input,
                labels=labels,
            )

            loss = output.loss

            results_loss_list.append(
                loss.cpu().item(),
            )
    elif mlm_pseudoperplexity_mode == MLMPseudoperplexityGranularity.TOKEN:
        for masked_input_row, labels_row in zip(masked_input, labels):
            masked_input_row = masked_input_row.unsqueeze(0)
            labels_row = labels_row.unsqueeze(0)

            with torch.inference_mode():
                output = model(
                    masked_input_row,
                    labels=labels_row,
                )

                loss = output.loss

                results_loss_list.append(
                    loss.cpu().item(),
                )
    else:
        msg = "Invalid value for `mlm_pseudoperplexity_mode`."
        raise ValueError(msg)

    results_loss_list_with_start_and_end = [0.0] + results_loss_list + [0.0]

    sentence_perplexity_container = SentencePerplexityContainer(
        token_ids=token_id_list,
        token_strings=tensor_input_decoded,
        token_perplexities=results_loss_list_with_start_and_end,
    )

    return sentence_perplexity_container


def token_level_to_sentence_level_pseudoperplexity(
    loss: torch.Tensor,
):
    return np.exp(loss.item())


def compute_perplexity_over_dataset(
    loaded_model_container: LoadedModelContainer,
    dataset: datasets.Dataset,
    column_name: str,
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> PerplexityResultsList:
    if loaded_model_container.lm_mode == LMmode.CLM:
        msg = "Perplexity computation not implemented for CLM yet."
        raise NotImplementedError(msg)

    results_list: PerplexityResultsList = []

    for index, single_entry in enumerate(
        tqdm(
            dataset,
            desc="Iterating over dataset",
        ),
    ):
        if not isinstance(
            single_entry,
            dict,
        ):
            msg = "Expected a dictionary."
            raise TypeError(msg)

        # Extract the sentence we want to compute the perplexity for.
        sentence = single_entry[column_name]

        result: SentencePerplexityContainer = pseudoperplexity_per_token_of_sentence(
            sentence=sentence,
            tokenizer=loaded_model_container.tokenizer,
            tokenizer_config=loaded_model_container.tokenizer_config,
            model=loaded_model_container.model,
            mlm_pseudoperplexity_mode=MLMPseudoperplexityGranularity.TOKEN,
            device=loaded_model_container.device,
            verbosity=verbosity,
            logger=logger,
        )

        results_list.append(
            (index, result),
        )

    if verbosity >= Verbosity.NORMAL:
        logger.info(
            "len(results_list):\n%s",
            len(results_list),
        )

    return results_list
