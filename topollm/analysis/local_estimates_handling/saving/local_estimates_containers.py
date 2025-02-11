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

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from topollm.analysis.local_estimates_computation.constants import (
    APPROXIMATE_HAUSDORFF_VIA_KDTREE_DICT_KEY,
)
from topollm.typing.enums import Verbosity

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)


def summarize_value(
    value: Any,
    key: str | None = None,
    fallback_truncation_length: int = 100,
) -> str:
    """Summarize a value for logging purposes.

    This function provides a concise summary depending on the type of `value`.

    Args:
        value:
            The value to summarize.
        key:
            The key to use to describe the value.
        fallback_truncation_length:
            The maximum length of the string representation of the value if no other summary is available.

    Returns:
        str: A summary string describing the value.

    """
    key_str: str = f"{key = }: " if key is not None else ""

    if value is None:
        value_str: str = "None"
    elif isinstance(
        value,
        np.ndarray,
    ):
        value_str = f"NumPy array with {value.shape = } and dtype {value.dtype}"
    elif isinstance(
        value,
        pd.DataFrame,
    ):
        value_str = f"DataFrame with {value.shape = } and columns {list(value.columns)}"
    elif isinstance(
        value,
        dict,
    ):
        value_str = f"Dict with keys {list(value.keys())}"
    else:
        value_str = str(object=value)[:fallback_truncation_length]

    result: str = key_str + value_str

    return result


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

    def log_info(
        self,
        verbosity: Verbosity = Verbosity.NORMAL,
        logger: logging.Logger = default_logger,
    ) -> None:
        """Log detailed information about this LocalEstimatesContainer instance.

        Each attribute is summarized rather than dumping the entire content, which
        is particularly useful for large arrays or data frames.
        """
        logger.info(
            msg="Logging LocalEstimatesContainer details:",
        )
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
            logger.info(
                msg=f"\t{summary}",  # noqa: G004 - low overhead
            )
