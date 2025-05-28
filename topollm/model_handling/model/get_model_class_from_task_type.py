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


"""Get the model class based on the task type."""

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
from transformers.modeling_utils import PreTrainedModel

from topollm.typing.enums import TaskType

TASK_TYPE_TO_MODEL_MAPPING = {
    TaskType.SEQUENCE_CLASSIFICATION: AutoModelForSequenceClassification,
    TaskType.TOKEN_CLASSIFICATION: AutoModelForTokenClassification,
    TaskType.MASKED_LM: AutoModelForMaskedLM,
    TaskType.CAUSAL_LM: AutoModelForCausalLM,
}


def get_model_class_from_task_type(
    task_type: TaskType,
) -> type[PreTrainedModel]:
    """Get the model class based on the task type."""
    return TASK_TYPE_TO_MODEL_MAPPING.get(
        task_type,
        AutoModel,
    )
