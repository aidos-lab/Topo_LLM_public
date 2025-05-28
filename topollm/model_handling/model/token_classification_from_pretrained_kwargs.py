# Copyright 2024
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


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
