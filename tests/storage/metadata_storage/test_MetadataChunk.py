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

import logging

import torch
from hypothesis import Verbosity, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import dictionaries

from topollm.storage.metadata_storage.MetadataChunk import MetadataChunk
from topollm.storage.StorageDataclasses import ChunkIdentifier

logger = logging.getLogger(__name__)


def tensors(
    dtype=torch.float32,
    min_dim=1,
    max_dim=3,
):
    # Generates dimensions for the tensor, within the specified range.
    # Each dimension will have a size chosen randomly between 1 and 10.
    dimensions = st.lists(
        st.integers(min_value=1, max_value=10),
        min_size=min_dim,
        max_size=max_dim,
    )

    # Generate tensors with the specified dimensions and dtype.
    return dimensions.map(
        lambda dims: torch.rand(
            *dims,
            dtype=dtype,
        ),
    )


chunk_identifier_strategy = st.builds(
    ChunkIdentifier,
    chunk_idx=st.integers(),
    start_idx=st.integers(),
    chunk_length=st.integers(),
)

metadata_chunk_strategy = st.builds(
    MetadataChunk,
    batch=dictionaries(
        keys=st.text(),
        values=tensors(),
    ).map(dict),
    chunk_identifier=chunk_identifier_strategy,
)


@given(
    metadata_chunk_strategy,
    metadata_chunk_strategy,
)
@settings(
    verbosity=Verbosity.verbose,
)
def test_metadata_chunk_equality(
    chunk1: MetadataChunk,
    chunk2: MetadataChunk,
):
    """
    Test that MetadataChunk equality works as expected
    """
    # Note: This logging produces a lot of output when run with Hypothesis,
    # because it will print all the generated examples.
    #
    # logger.info(f"{chunk1 = }")
    # logger.info(f"{chunk2 = }")

    # Check reflexivity
    assert chunk1 == chunk1

    # Check symmetry
    assert (chunk1 == chunk2) == (chunk2 == chunk1)

    # Check cases where they should not be equal
    if chunk1.chunk_identifier != chunk2.chunk_identifier or chunk1.batch.keys() != chunk2.batch.keys():
        assert chunk1 != chunk2
