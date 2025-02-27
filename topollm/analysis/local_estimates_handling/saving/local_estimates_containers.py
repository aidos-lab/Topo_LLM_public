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

"""Container for the local estimates data."""

import dataclasses
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from topollm.analysis.local_estimates_computation.constants import (
    APPROXIMATE_HAUSDORFF_VIA_KDTREE_DICT_KEY,
)
from topollm.logging.summarize_value import summarize_value
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


@dataclass
class LocalEstimatesContainer:
    """Container for the local estimates data."""

    # Required: Array with the pointwise results
    pointwise_results_array_np: np.ndarray
    pointwise_results_meta_frame: pd.DataFrame | None = None

    # Optional: Array with the global estimate
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

    def get_pointwise_results_np_mean(
        self,
    ) -> float:
        if self.pointwise_results_array_np is None:
            msg = "No pointwise results available."
            raise ValueError(
                msg,
            )

        output = np.mean(
            a=self.pointwise_results_array_np,
            axis=0,
        )

        return output

    def get_pointwise_results_np_std(
        self,
    ) -> float:
        if self.pointwise_results_array_np is None:
            msg = "No pointwise results available."
            raise ValueError(
                msg,
            )

        output = np.std(
            a=self.pointwise_results_array_np,
            axis=0,
        )

        return output

    def get_approximate_hausdorff_via_kd_tree(
        self,
    ) -> float:
        if self.additional_distance_computations_results is None:
            msg = "No additional distance computations results available."
            raise ValueError(
                msg,
            )

        if APPROXIMATE_HAUSDORFF_VIA_KDTREE_DICT_KEY not in self.additional_distance_computations_results:
            msg = "No approximate Hausdorff via KDTree distance available."
            raise ValueError(
                msg,
            )

        output = self.additional_distance_computations_results[APPROXIMATE_HAUSDORFF_VIA_KDTREE_DICT_KEY]

        return output

    def get_summary_string(
        self,
        indent: str = "\t",
    ) -> str:
        """Generate a summary string for all fields of the container with indentation.

        Each attribute is summarized rather than dumping the entire content,
        which is particularly useful for large arrays or data frames.

        Args:
            indent:
                The string to prefix each summary line for the fields.

        Returns:
            str: A summary string starting with a header and followed by a newline-separated
                 list of each field's summary.

        """
        summary_lines: list[str] = ["LocalEstimatesContainer details:"]
        for field_info in dataclasses.fields(
            class_or_instance=self,
        ):
            value = getattr(
                self,
                field_info.name,
            )
            summary = summarize_value(
                value=value,
                key=field_info.name,
            )
            summary_lines.append(f"{indent}{summary}")
        return "\n".join(summary_lines)
