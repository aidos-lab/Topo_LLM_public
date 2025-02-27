import logging
import pathlib
import sys

import numpy as np
import torch
from erc_models import ContextBERT_ERToD

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH
from topollm.logging.log_model_info import log_model_info

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
# Set the logger level to INFO and add a StreamHandler to the logger
default_logger.setLevel(
    level=logging.INFO,
)
formatter = logging.Formatter(
    fmt="[%(asctime)s][%(levelname)8s][%(name)s] %(message)s (%(filename)s:%(lineno)s)",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(fmt=formatter)
default_logger.addHandler(hdlr=handler)


def main() -> None:
    logger = default_logger

    logger.info(
        msg=f"{TOPO_LLM_REPOSITORY_BASE_PATH=}",  # noqa: G004 - low overhead
    )

    # ======================================================== #
    # File paths
    # ======================================================== #

    required_files_dst_dir = pathlib.Path(
        TOPO_LLM_REPOSITORY_BASE_PATH,
        "data/models/EmoLoop/required_files/dst",
    )

    bert_base_model_path = pathlib.Path(
        required_files_dst_dir,
        "bert-base-uncased",
    )
    erc_state_dict_path = pathlib.Path(
        required_files_dst_dir,
        "contextbert-ertod.pt",
    )

    # This is where the BERT-component of the ERC model will be saved
    bert_component_save_path = pathlib.Path(
        TOPO_LLM_REPOSITORY_BASE_PATH,
        "data/models/EmoLoop/ContextBERT_ERToD/extracted_bert_component/checkpoint-best",
    )

    # ======================================================== #
    # Load the ERC model
    # ======================================================== #

    erc_model: ContextBERT_ERToD = load_contextbert_ertod_model(
        bert_base_model_path=bert_base_model_path,
        erc_state_dict_path=erc_state_dict_path,
    )

    # ======================================================== #
    # Extract the BERT-based component of the ERC model
    # and save as a separate model to disk.
    # ======================================================== #

    bert_component = erc_model.bert

    log_model_info(
        model=bert_component,
        model_name="bert_component",
        logger=logger,
    )

    bert_component_save_path.mkdir(
        parents=True,
        exist_ok=True,
    )
    # bert_component is a transformer model,
    # so we can save it using the save_pretrained method
    bert_component.save_pretrained(
        save_directory=bert_component_save_path,
    )

    # ======================================================== #
    # Using the ERC model.
    # ======================================================== #

    # # # #
    # Optional: Example usage of the model prediction
    run_example_forward_pass_on_erc_model(
        erc_model=erc_model,
    )


def run_example_forward_pass_on_erc_model(
    erc_model: ContextBERT_ERToD,
) -> None:
    """Run an example forward pass on the ERC model."""
    history: list[str] = [
        "hey. I need a cheap restaurant.",
        "There are many cheap places, which food do you like?",
    ]
    user_utt = "If you have something Asian that would be great. I am very excited."

    e = predict_with_dummy_ds(
        erc_model,
        user_utt,
        history,
    )
    print(e)


def load_contextbert_ertod_model(
    bert_base_model_path: pathlib.Path,
    erc_state_dict_path: pathlib.Path,
) -> ContextBERT_ERToD:
    """Load the ContextBERT_ERToD model from the specified paths."""
    erc_model = ContextBERT_ERToD(
        base_model_path=str(bert_base_model_path),
    )

    # TODO: Check if the `strict=False` might be a problem here, because the positional parameters might not have been loaded.
    if torch.cuda.is_available():
        erc_model.load_state_dict(
            state_dict=torch.load(erc_state_dict_path)["state_dict"],
            strict=False,
        )
    else:
        erc_model.load_state_dict(
            state_dict=torch.load(
                erc_state_dict_path,
                map_location=torch.device(
                    "cpu",
                ),
            )["state_dict"],
            strict=False,
        )

    return erc_model


def predict_with_dummy_ds(
    model,
    user_utt,
    history,
):
    # TODO: Check if the different training data formatting might be the problem here.
    history_str = ""
    for i in reversed(range(len(history))):  # reverse order to place the current turn closer to the [CLS]
        if i % 2 == 0:
            history_str += f"user: {history[i]} "
        else:
            history_str += f"system: {history[i]} "
    text = f"user: {user_utt} {history_str}"

    encoding = model.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=model.max_token_len,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True,
    )

    ctx_ds_len = model.ds_ctx_window_size * model.use_dialog_state
    ctx_ds_vec = [np.zeros((model.ds_dim,))] * ctx_ds_len
    ctx_ds_vec = np.concatenate(tuple(ctx_ds_vec[::-1][:ctx_ds_len]), axis=None)

    input_ids = torch.LongTensor(encoding["input_ids"])
    attention_mask = torch.LongTensor(encoding["attention_mask"])
    dialog_state_vec = torch.FloatTensor(ctx_ds_vec).unsqueeze(0)
    print(dialog_state_vec.shape)

    with torch.no_grad():
        emotion_logits, _, _, _ = model.forward(input_ids, attention_mask, dialog_state_vec)
    _, emo_pred = torch.max(emotion_logits, dim=1)

    return emo_pred


if __name__ == "__main__":
    main()
