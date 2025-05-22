# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Matthias Ruppik (mail@ruppik.net)
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

from typing import Any

import torch


def move_batch_to_cpu(
    batch: dict,
) -> dict:
    """Move all tensors in the batch to CPU, including nested dictionaries."""

    def recursive_to_cpu(
        data: Any,  # noqa: ANN401 - Any type is necessary here
    ) -> Any:  # noqa: ANN401 - Any type is necessary here
        if isinstance(
            data,
            torch.Tensor,
        ):
            return data.cpu()
        elif isinstance(
            data,
            dict,
        ):
            return {key: recursive_to_cpu(value) for key, value in data.items()}
        elif isinstance(
            data,
            list,
        ):
            return [recursive_to_cpu(item) for item in data]
        elif isinstance(
            data,
            tuple,
        ):
            return tuple(recursive_to_cpu(item) for item in data)
        # Add other structures if necessary
        return data

    return recursive_to_cpu(
        data=batch,
    )
