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
import sklearn.decomposition
import sklearn.manifold


def create_projected_data(
    array: np.ndarray,
    pca_n_components: int | None = 50,
    tsne_n_components: int = 2,
    tsne_random_state: int = 42,
) -> np.ndarray:
    """Create a projection of the input data using PCA and t-SNE."""
    # Apply PCA if requested
    if pca_n_components:
        pca = sklearn.decomposition.PCA(
            n_components=pca_n_components,
        )
        array = pca.fit_transform(
            array,
        )

    # Apply t-SNE
    tsne = sklearn.manifold.TSNE(
        n_components=tsne_n_components,
        random_state=tsne_random_state,
    )
    tsne_array = tsne.fit_transform(
        array,
    )

    return tsne_array
