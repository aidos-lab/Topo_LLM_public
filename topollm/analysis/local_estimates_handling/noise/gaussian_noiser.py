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

import logging

import numpy as np
import pandas as pd

from topollm.analysis.local_estimates_handling.noise.gaussian_distortion import add_gaussian_distortion
from topollm.embeddings_data_prep.prepared_data_containers import PreparedData
from topollm.logging.log_array_info import log_array_info
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


class GaussianNoiser:
    """Remove duplicate vectors and corresponding metadata from the prepared data."""

    def __init__(
        self,
        distortion_param: float,
        seed: int,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Initialize the deduplicator."""
        self.distortion_param: float = distortion_param
        self.seed: int = seed

        self.verbosity: Verbosity = verbosity
        self.logger: logging.Logger = logger

    def apply_noise_to_data(
        self,
        prepared_data: PreparedData,
    ) -> PreparedData:
        """Applay numpy.unique function to the array and align metadata."""
        input_array = prepared_data.array
        input_meta_frame = prepared_data.meta_df

        distorted_array = add_gaussian_distortion(
            original_array=input_array,
            distortion_param=self.distortion_param,
            seed=self.seed,
        )

        output_data = PreparedData(
            array=distorted_array,
            meta_df=input_meta_frame,
        )

        return output_data
