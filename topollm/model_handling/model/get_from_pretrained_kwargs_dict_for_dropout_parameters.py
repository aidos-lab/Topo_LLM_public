from topollm.config_classes.language_model.language_model_config import LanguageModelConfig
from topollm.typing.enums import DropoutMode


def get_from_pretrained_kwargs_dict_for_dropout_parameters(
    language_model_config: LanguageModelConfig,
) -> dict:
    """Get the keyword arguments for the dropout parameters."""
    match language_model_config.dropout.mode:
        case DropoutMode.DEFAULTS:
            # Empty dictionary for default values.
            from_pretrained_kwargs_dict: dict = {}
        case DropoutMode.MODIFY_ROBERTA_DROPOUT_PARAMETERS:
            from_pretrained_kwargs_dict: dict = {
                "hidden_dropout_prob": language_model_config.dropout.probabilities.hidden_dropout_prob,
                "attention_probs_dropout_prob": language_model_config.dropout.probabilities.attention_probs_dropout_prob,
                "classifier_dropout": language_model_config.dropout.probabilities.classifier_dropout,
            }
        # Note: We can extend this switch statement with additional cases for other dropout modes
        # if we add support for other language models later.
        case _:
            msg: str = f"Unknown {language_model_config.dropout.mode = }"
            raise ValueError(
                msg,
            )

    return from_pretrained_kwargs_dict
