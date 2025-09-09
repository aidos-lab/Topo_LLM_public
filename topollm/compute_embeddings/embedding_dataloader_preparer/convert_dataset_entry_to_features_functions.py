"""Functions for converting dataset entries to features."""

from collections.abc import Callable

import nltk
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerFast

from topollm.config_classes.data.data_config import DataConfig
from topollm.typing.enums import DatasetType


def get_convert_dataset_entry_to_features_function(
    data_config: DataConfig,
) -> Callable[
    ...,
    BatchEncoding,
]:
    """Get the function to convert a dataset entry to features."""
    match data_config.dataset_type:
        case DatasetType.HUGGINGFACE_DATASET:
            dataset_entry_to_features_function: Callable = convert_dataset_entry_to_features
        case DatasetType.HUGGINGFACE_DATASET_NAMED_ENTITY:
            dataset_entry_to_features_function: Callable = convert_dataset_entry_to_features_named_entity
        case (
            DatasetType.HUGGINGFACE_DATASET_PRETOKENIZED
            | DatasetType.SETSUMBT_DATALOADERS_PROCESSED
            | DatasetType.TRIPPY_DATALOADERS_PROCESSED
            | DatasetType.TRIPPY_R_DATALOADERS_PROCESSED
        ):
            # In this mode, the dataset entries are already tokenized and can be directly used as features.
            dataset_entry_to_features_function: Callable = convert_dataset_entry_to_features_do_nothing
        case _:
            msg: str = f"Unsupported {data_config.dataset_type = }"
            raise ValueError(msg)

    return dataset_entry_to_features_function


def convert_dataset_entry_to_features_do_nothing(
    dataset_entry: dict,
    _tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    _column_name: str = "text",
    _max_length: int = 512,
) -> BatchEncoding:
    """Do nothing.

    Note that the arguments `_tokenizer`, `_column_name`, and `_max_length` are not used,
    but are included to match the signature of other conversion functions.
    """
    return BatchEncoding(
        data=dataset_entry,
    )


def convert_dataset_entry_to_features(
    dataset_entry: dict[str, list],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    column_name: str = "text",
    max_length: int = 512,
) -> BatchEncoding:
    """Convert dataset entries to features by tokenizing the text and padding/truncating to a maximum length."""
    features: BatchEncoding = tokenizer(
        dataset_entry[column_name],
        max_length=max_length,
        padding="max_length",
        truncation="longest_first",
    )

    return features


def convert_dataset_entry_to_features_named_entity(
    dataset_entry: dict,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    column_name: str = "text",
    max_length: int = 512,
    pos_tags_name: str = "POS",
) -> BatchEncoding:
    """Convert dataset entries to features by tokenizing the text and padding/truncating to a maximum length.

    Additionally, compute part-of-speech (POS) tags for each token and include them in the features.
    """
    split_words: list[list[str]] = [
        nltk.word_tokenize(
            text=sent,
        )
        for sent in dataset_entry[column_name]
    ]

    features: BatchEncoding = tokenizer(
        split_words,
        max_length=max_length,
        padding="max_length",
        truncation="longest_first",
        is_split_into_words=True,
    )

    word_ids: list[list[int | None]] = [features.word_ids(batch_index=i) for i in range(len(split_words))]

    dataset_tokenized = features.input_ids

    pos_tag = [nltk.pos_tag(tokens=sent) for sent in split_words]

    all_word_tags_one_sentence_tokens = []

    for sentence_idx in range(len(dataset_tokenized)):
        word_tags_one_sentence = pos_tag[sentence_idx]
        word_tags_one_sentence = [word_tags_one_sentence[i][1] for i in range(len(word_tags_one_sentence))]
        word_ids_one_sentence = word_ids[sentence_idx]

        word_tags_one_sentence_tokens: list = []
        for i in word_ids_one_sentence:
            if i is not None:
                word_tags_one_sentence_tokens.append(word_tags_one_sentence[i])
            else:
                word_tags_one_sentence_tokens.append(None)
        all_word_tags_one_sentence_tokens.append(word_tags_one_sentence_tokens)

    features[pos_tags_name] = all_word_tags_one_sentence_tokens

    return features
