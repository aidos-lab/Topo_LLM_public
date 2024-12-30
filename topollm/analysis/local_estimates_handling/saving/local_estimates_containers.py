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

"""Container for the local estimates data."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LocalEstimatesContainer:
    """Container for the local estimates data."""

    pointwise_results_array_np: np.ndarray
    pointwise_results_meta_frame: pd.DataFrame | None = None
    global_estimate_array_np: np.ndarray | None = None

    # Optional: Array which was used to compute the local estimates
    array_for_estimator_np: np.ndarray | None = None

    # Optional additional results
    additional_distance_computations_results: dict | None = None
    additional_pointwise_results_statistics: dict | None = None

    def get_global_estimate(
        self,
    ) -> float:
        if self.global_estimate_array_np is None:
            msg = "No global estimate available."
            raise ValueError(
                msg,
            )

        output = self.global_estimate_array_np[0]

        return output
