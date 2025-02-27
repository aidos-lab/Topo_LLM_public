import logging
import pathlib
import sys

import numpy as np
import torch
from erc_models import ContextBERT_ERToD

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH

default_logger: logging.Logger = logging.getLogger(
    name=__name__,
)
# Set the logger level to INFO and add a StreamHandler to the logger
default_logger.setLevel(
    level=logging.INFO,
)
formatter = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(fmt=formatter)
default_logger.addHandler(hdlr=handler)


def predict_with_dummy_ds(
    model,
    user_utt,
    history,
):
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


def main() -> None:
    logger = default_logger

    logger.info(
        msg=f"{TOPO_LLM_REPOSITORY_BASE_PATH=}",  # noqa: G004 - low overhead
    )

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

    erc_model: ContextBERT_ERToD = load_contextbert_ertod_model(
        bert_base_model_path=bert_base_model_path,
        erc_state_dict_path=erc_state_dict_path,
    )

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


if __name__ == "__main__":
    main()
