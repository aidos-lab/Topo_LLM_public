# coding=utf-8
#
# Copyright 2024
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (ruppik@hhu.de)
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
import random
import logging

import torch
import torch.backends.cudnn


def set_seed(
    seed: int,
    logger: logging.Logger = logging.getLogger(__name__),
) -> None:
    """
    Sets the seed for generating random numbers in PyTorch and numpy.

    Args:
        seed (int): The seed for the random number generator.

    Notes:
        1. The RNG state for the CUDA is set, which makes CUDA operations deterministic.
        2. A seed for the Python built-in random module is also set.
        3. PyTorch's cuDNN uses nondeterministic algorithms which can be
           disabled setting `torch.backends.cudnn.deterministic = True`.
           However, this can slow down the computations.
        4. PyTorch's cuDNN has a benchmark mode which allows hardware
           optimizations for the operations. This can be enabled or disabled
           using `torch.backends.cudnn.benchmark`. Disabling it helps in making
           the computations deterministic.
        5. For operations performed on CPU and CUDA, setting the seed ensures
           reproducibility across multiple runs.
    """
    random.seed(seed)  # Set the seed for Python's built-in random module

    np.random.seed(seed)  # Set the seed for numpy random number generator

    torch.manual_seed(seed)  # Set the seed for CPU operations

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Set the seed for all GPU devices

    torch.backends.cudnn.deterministic = True  # Disable nondeterministic algorithms
    torch.backends.cudnn.benchmark = False  # Disable hardware optimizations

    logger.info(f"seed set to {seed = }.")

    return None
