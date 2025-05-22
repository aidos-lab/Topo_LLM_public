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

"""Container for prepared data."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from topollm.logging.log_array_info import log_array_info
from topollm.logging.log_dataframe_info import log_dataframe_info

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


@dataclass
class PreparedData:
    """Container for prepared data."""

    array: np.ndarray
    meta_df: pd.DataFrame

    def log_info(
        self,
        logger: logging.Logger = default_logger,
    ) -> None:
        log_array_info(
            array_=self.array,
            array_name="array",
            logger=logger,
        )
        log_dataframe_info(
            df=self.meta_df,
            df_name="meta_df",
            logger=logger,
        )
