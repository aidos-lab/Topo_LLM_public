"""Dataset entry -> feature conversion utilities and mask visualization.

Provides conversion helpers for different dataset types.
"""

from collections.abc import Callable

import nltk
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerFast

from topollm.compute_embeddings.embedding_dataloader_preparer.extract_spans import (
    _find_last_label_span,
    _mask_from_span,
)
from topollm.config_classes.data.data_config import DataConfig
from topollm.typing.enums import DatasetType


def get_convert_dataset_entry_to_features_function(
    data_config: DataConfig,
) -> Callable[..., BatchEncoding]:
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
        case DatasetType.LUSTER_DATASET:
            dataset_entry_to_features_function: Callable = convert_dataset_entry_to_features_luster_data
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


def build_basic_segment_masks_for_encoded_text(
    texts: list[str],
    encodings: BatchEncoding,
    delimiter: str = "</s>",
) -> dict[str, list[list[int]]]:
    """Tokenize and build masks for the last system, state, and database segments.

    Args:
        texts: List of full dataset entry strings.
        encodings: BatchEncoding with offset_mapping and special_tokens_mask.
        delimiter: Segment delimiter.

    Returns:
        masks: dict with keys:
            'mask_system_last', 'mask_state', 'mask_database'
            Each value is a list of masks (one per example).

    """
    offsets_batch: list[list[tuple[int, int]]] = encodings.offset_mapping
    specials_batch: list[list[int]] = encodings.special_tokens_mask

    out: dict[str, list[list[int]]] = {
        "mask_system_last": [],
        "mask_state": [],
        "mask_database": [],
        "mask_action": [],
    }

    for (
        text,
        offsets,
        specials,
    ) in zip(
        texts,
        offsets_batch,
        specials_batch,
        strict=True,
    ):
        span_sys: tuple[int, int] | None = _find_last_label_span(
            text=text,
            label="system",
            delimiter=delimiter,
            include_marker=False,
        )
        span_state: tuple[int, int] | None = _find_last_label_span(
            text=text,
            label="state",
            delimiter=delimiter,
            include_marker=False,
        )
        span_db: tuple[int, int] | None = _find_last_label_span(
            text=text,
            label="database",
            delimiter=delimiter,
            include_marker=False,
        )
        span_action: tuple[int, int] | None = _find_last_label_span(
            text=text,
            label="action",
            delimiter=delimiter,
            include_marker=False,
        )

        out["mask_system_last"].append(_mask_from_span(offsets=offsets, span=span_sys, special_tokens_mask=specials))
        out["mask_state"].append(_mask_from_span(offsets=offsets, span=span_state, special_tokens_mask=specials))
        out["mask_database"].append(_mask_from_span(offsets=offsets, span=span_db, special_tokens_mask=specials))
        out["mask_action"].append(_mask_from_span(offsets=offsets, span=span_action, special_tokens_mask=specials))

    return out


def convert_dataset_entry_to_features_luster_data(
    dataset_entry: dict[str, list],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    column_name: str = "source_target",
    max_length: int = 512,
) -> BatchEncoding:
    """Convert dataset entries to features specific for LUSTER data.

    In addition to the tokenization, this function creates specific masks to highlight certain parts of the input text.
    """
    tokenized_entries: BatchEncoding = tokenizer(
        dataset_entry[column_name],
        max_length=max_length,
        padding="max_length",
        truncation="longest_first",
        return_offsets_mapping=True,  # return list of (start_char, end_char) per token
        return_special_tokens_mask=True,
    )

    segment_masks: dict[str, list[list[int]]] = build_basic_segment_masks_for_encoded_text(
        texts=dataset_entry[column_name],
        encodings=tokenized_entries,
        delimiter="</s>",
    )

    # Combine the tokenized entries with the segment masks
    features = BatchEncoding(
        data={
            **tokenized_entries,
            **segment_masks,
        },
        encoding=tokenized_entries.encodings,
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

    pos_tag: list[list] = [nltk.pos_tag(tokens=sent) for sent in split_words]

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
