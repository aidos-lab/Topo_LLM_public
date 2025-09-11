"""Functions for converting dataset entries to features."""

import re
import warnings
from collections.abc import Callable, Iterable

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


def _find_last_label_span(
    text: str,
    label: str,
    delimiter: str = "</s>",
    *,
    include_marker: bool = False,
    case_sensitive: bool = False,
) -> tuple[int, int] | None:
    """Find the [start, end) character span of the last '<label> : ... </s>' segment.

    Args:
      text: Full dataset entry string.
      label: Segment label, e.g., 'system', 'state', 'database'.
      delimiter: End-of-segment delimiter.
      include_marker: If True, include the leading 'label :' in the span.
      case_sensitive: Match case-sensitively if True.

    Returns:
      A 2-tuple (start, end) in character indices, or None if not found.

    """
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(rf"{re.escape(label)}\s*:\s*", flags)
    last = None

    for m in pattern.finditer(text):
        marker_start, marker_end = m.start(), m.end()
        seg_end = text.find(delimiter, marker_end)
        if seg_end == -1:
            seg_end = len(text)
        span_start = marker_start if include_marker else marker_end
        span_end = seg_end
        last = (span_start, span_end)

    return last


def _mask_from_span(
    offsets: list[tuple[int, int]],
    span: tuple[int, int] | None,
    special_tokens_mask: list[int] | None = None,
) -> list[int]:
    """Build a 0/1 mask for tokens whose char offsets intersect the given span.

    Args:
      offsets: (start, end) per token. (start==end) often indicates special/pad.
      span: (a, b) character span in the same text; None -> all zeros.
      special_tokens_mask: If provided, any token with value==1 is zeroed.

    Returns:
      A list of 0/1 ints with same length as offsets.

    """
    mask = [0] * len(offsets)
    if span is None:
        return mask

    a, b = span
    for i, (s, e) in enumerate(offsets):
        # Skip special/pad encoded as (0,0) by some tokenizers
        if s == 0 and e == 0:
            continue
        # Overlap test for half-open intervals
        if s < b and e > a:
            mask[i] = 1

    if special_tokens_mask is not None:
        mask = [m if special_tokens_mask[i] == 0 else 0 for i, m in enumerate(mask)]

    return mask


def build_basic_segment_masks(
    texts: list[str],
    encodings: BatchEncoding,
    delimiter: str = "</s>",
) -> dict[str, list[list[int]]]:
    """Tokenize and build masks for the last system, state, and database segments.

    Args:
      delimiter: Segment delimiter.

    Returns:
      (encoding, masks) where:
        - encoding: BatchEncoding with input_ids, offset_mapping, etc.
        - masks: dict with keys:
            'mask_system_last', 'mask_state', 'mask_database'
          Each value is a list of masks (one per example).

    """
    offsets_batch: list[list[tuple[int, int]]] = encodings.offset_mapping
    specials_batch: list[list[int]] = encodings.special_tokens_mask

    out: dict[str, list[list[int]]] = {
        "mask_system_last": [],
        "mask_state": [],
        "mask_database": [],
    }

    for text, offsets, specials in zip(
        texts,
        offsets_batch,
        specials_batch,
        strict=True,
    ):
        span_sys = _find_last_label_span(
            text,
            "system",
            delimiter=delimiter,
            include_marker=False,
        )
        span_state = _find_last_label_span(
            text,
            "state",
            delimiter=delimiter,
            include_marker=False,
        )
        span_db = _find_last_label_span(
            text,
            "database",
            delimiter=delimiter,
            include_marker=False,
        )

        out["mask_system_last"].append(_mask_from_span(offsets, span_sys, specials))
        out["mask_state"].append(_mask_from_span(offsets, span_state, specials))
        out["mask_database"].append(_mask_from_span(offsets, span_db, specials))

    return out


def debug_str_masks(
    features: BatchEncoding,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    mask_keys: list[str] | None = None,
) -> str:
    """Pretty-print tokens and aligned masks for manual inspection.

    Args:
      features: BatchEncoding with input_ids, offset_mapping, etc.
      tokenizer: HF tokenizer.
      mask_keys: If provided, only include these masks; otherwise all masks in features.

    """
    if mask_keys is None:
        mask_keys = [k for k in features if re.match(r"^mask_", k)]

    def fmt(row: Iterable[int | str]) -> str:
        return " ".join(str(x)[:cell].ljust(cell) for x in row)

    output_lines: list[str] = []

    for index in range(len(features.input_ids)):
        cell = 12
        lines = []

        tokens = tokenizer.convert_ids_to_tokens(features.input_ids[index])
        lines.append(fmt(["Token", *tokens]))

        for key in mask_keys:
            if key in features:
                mask = features[key][index]  # type: ignore - this access via key is valid for BatchEncoding objects
                lines.append(fmt([key, *mask]))

        output_lines.append("\n".join(lines))

    return "\n".join(output_lines)


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

    segment_masks: dict[str, list[list[int]]] = build_basic_segment_masks(
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
