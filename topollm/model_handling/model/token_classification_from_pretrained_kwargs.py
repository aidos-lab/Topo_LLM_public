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

from pydantic import BaseModel, Field

label_list_default = [
    "O",
    "B-corporation",
    "I-corporation",
    "B-creative-work",
    "I-creative-work",
    "B-group",
    "I-group",
    "B-location",
    "I-location",
    "B-person",
    "I-person",
    "B-product",
    "I-product",
]
id2label_default = dict(enumerate(label_list_default))
label2id_default = {label: i for i, label in enumerate(label_list_default)}


class TokenClassificationFromPretrainedKwargs(BaseModel):
    """Token classification from pretrained kwargs.

    NOTE: Only add those fields here which will be consumed by the model
    `.from_pretrained()` method.
    In particular, do NOT add the `label_list` field here.
    """

    # Note: Do NOT add the `label_list` field here.
    num_labels: int = Field(
        default=len(label_list_default),
        title="Number of labels",
    )
    id2label: dict[int, str] = Field(
        default=id2label_default,
        title="ID to label mapping",
    )

    label2id: dict[str, int] = Field(
        default=label2id_default,
        title="Label to ID mapping",
    )
