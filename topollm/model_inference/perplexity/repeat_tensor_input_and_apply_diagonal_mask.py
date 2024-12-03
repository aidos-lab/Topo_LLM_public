# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (mail@ruppik.net)
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

import torch

default_device: torch.device = torch.device(
    device="cpu",
)


def repeat_tensor_input_and_apply_diagonal_mask(
    tensor_input: torch.Tensor,
    mask_token_id: int,
    device: torch.device = default_device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """Repeat the input tensor and apply a diagonal mask to it.

    This function expects a tensor with shape `(1, sequence_length)`.
    It also expects that there are special tokens at the start and end of the sequence,
    and these will be carried along in the output tensor
    but the mask will be applied only to the non-special tokens in the middle.
    """
    # Example:
    # `model = 'roberta-base'`
    # `sentence = 'Paris is in France.'
    # `tensor_input = tensor([[    0, 32826,    16,    11,  1470,     4,     2]])`
    # `[tokenizer.decode(single_token_id) for single_token_id in tensor_input[0]]
    #  = ['<s>', 'Paris', ' is', ' in', ' France', '.', '</s>']`

    repeat_input: torch.Tensor = tensor_input.repeat(
        tensor_input.size(dim=-1) - 2,
        1,
    )
    # > repeat_input =
    # > tensor([[    0, 32826,    16,    11,  1470,     4,     2],
    # >         [    0, 32826,    16,    11,  1470,     4,     2],
    # >         [    0, 32826,    16,    11,  1470,     4,     2],
    # >         [    0, 32826,    16,    11,  1470,     4,     2],
    # >         [    0, 32826,    16,    11,  1470,     4,     2]])

    diagonal_mask: torch.Tensor = torch.ones(
        tensor_input.size(dim=-1) - 1,
        device=device,
    ).diag(
        diagonal=1,
    )[:-2]
    # > diagonal_mask =
    # > tensor([[0., 1., 0., 0., 0., 0., 0.],
    # >         [0., 0., 1., 0., 0., 0., 0.],
    # >         [0., 0., 0., 1., 0., 0., 0.],
    # >         [0., 0., 0., 0., 1., 0., 0.],
    # >         [0., 0., 0., 0., 0., 1., 0.]])

    masked_input: torch.Tensor = repeat_input.masked_fill(
        mask=(diagonal_mask == 1),
        value=mask_token_id,
    )
    # > masked_input =
    # > tensor([[    0, 50264,    16,    11,  1470,     4,     2],
    # >         [    0, 32826, 50264,    11,  1470,     4,     2],
    # >         [    0, 32826,    16, 50264,  1470,     4,     2],
    # >         [    0, 32826,    16,    11, 50264,     4,     2],
    # >         [    0, 32826,    16,    11,  1470, 50264,     2]])

    labels: torch.Tensor = repeat_input.masked_fill(
        mask=(masked_input != mask_token_id),
        value=-100,
    )
    # > labels =
    # > tensor([[ -100, 32826,  -100,  -100,  -100,  -100,  -100],
    # >         [ -100,  -100,    16,  -100,  -100,  -100,  -100],
    # >         [ -100,  -100,  -100,    11,  -100,  -100,  -100],
    # >         [ -100,  -100,  -100,  -100,  1470,  -100,  -100],
    # >         [ -100,  -100,  -100,  -100,  -100,     4,  -100]])

    return masked_input, labels
