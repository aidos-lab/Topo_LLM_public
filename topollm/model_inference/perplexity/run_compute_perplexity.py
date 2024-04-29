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

import numpy as np
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


def pseudoperplexity_per_token_of_sentence(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    model: PreTrainedModel,
    sentence: str,
) -> torch.Tensor:
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
    diagonal_mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(
        mask=(diagonal_mask == 1),
        value=mask_token_id,
    )
    labels = repeat_input.masked_fill(
        mask=(masked_input != mask_token_id),
        value=-100,
    )

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


def compute_perplexity():
    # TODO(Ben): Implement a script which computes (pseudo-)perplexity of a model on a given dataset.
    # TODO(Ben): Save the token-level (pseudo-)perplexity to an array.

    model_name = "cointegrated/rubert-tiny"
    model = AutoModelForMaskedLM.from_pretrained(
        model_name,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )

    # # # #
    result = pseudoperplexity_per_token_of_sentence(
        tokenizer=tokenizer,
        model=model,
        sentence="London is the capital of Great Britain.",
    )

    print(result)
    # 4.541251105675365

    # # # #
    result = pseudoperplexity_per_token_of_sentence(
        tokenizer=tokenizer,
        model=model,
        sentence="London is the capital of South America.",
    )
    print(result)
    # 6.162017238332462


def main() -> None:
    compute_perplexity()


if __name__ == "__main__":
    main()
