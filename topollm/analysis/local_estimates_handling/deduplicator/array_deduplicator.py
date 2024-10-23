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

from topollm.embeddings_data_prep.prepared_data_containers import PreparedData


class ArrayDeduplicator:
    """Remove duplicate vectors and corresponding metadata from the prepared data."""

    def filter_data(
        self,
        prepared_data: PreparedData,
    ) -> PreparedData:
        """Applay numpy.unique function to the array and align metadata."""
        input_array = prepared_data.array
        inpute_meta_frame = prepared_data.meta_df

        # TODO: Implement this function

        output_meta_frame = inpute_meta_frame.iloc[indices_to_keep]

        output_data = PreparedData(
            array=output_array,
            meta_df=output_meta_frame,
        )

        return output_data
