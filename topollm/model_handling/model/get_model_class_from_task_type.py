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
