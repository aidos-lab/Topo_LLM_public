"""Dataset entry -> feature conversion utilities and mask visualization.

Provides conversion helpers for different dataset types.
"""

import re
from collections.abc import Callable
from typing import Literal

import nltk
from rich import box
from rich.console import Console
from rich.table import Table
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerFast

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
    flags: Literal[0] | re.RegexFlag = 0 if case_sensitive else re.IGNORECASE
    pattern: re.Pattern[str] = re.compile(
        pattern=rf"{re.escape(pattern=label)}\s*:\s*",
        flags=flags,
    )
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
    mask: list[int] = [0] * len(offsets)
    if span is None:
        return mask

    (
        a,
        b,
    ) = span
    for i, (s, e) in enumerate(iterable=offsets):
        # Skip special/pad encoded as (0,0) by some tokenizers
        if s == 0 and e == 0:
            continue
        # Overlap test for half-open intervals
        if s < b and e > a:
            mask[i] = 1

    if special_tokens_mask is not None:
        mask = [m if special_tokens_mask[i] == 0 else 0 for i, m in enumerate(iterable=mask)]

    return mask


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

    for text, offsets, specials in zip(
        texts,
        offsets_batch,
        specials_batch,
        strict=True,
    ):
        span_sys = _find_last_label_span(
            text=text,
            label="system",
            delimiter=delimiter,
            include_marker=False,
        )
        span_state = _find_last_label_span(
            text=text,
            label="state",
            delimiter=delimiter,
            include_marker=False,
        )
        span_db = _find_last_label_span(
            text=text,
            label="database",
            delimiter=delimiter,
            include_marker=False,
        )
        span_action = _find_last_label_span(
            text=text,
            label="action",
            delimiter=delimiter,
            include_marker=False,
        )

        out["mask_system_last"].append(_mask_from_span(offsets, span_sys, specials))
        out["mask_state"].append(_mask_from_span(offsets, span_state, specials))
        out["mask_database"].append(_mask_from_span(offsets, span_db, specials))
        out["mask_action"].append(_mask_from_span(offsets, span_action, specials))

    return out


def debug_str_masks(
    features: BatchEncoding,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    mask_keys: list[str] | None = None,
    *,
    max_rows: int | None = None,
) -> str:
    """Return a rich table with one row per token and one column per mask.

    Layout:
        Columns: [Idx, Token, <mask_*> ...]
        Each mask column shows 0/1 with highlighted 1 values.

    Parameters
    ----------
    features : BatchEncoding
        Must include ``input_ids`` and mask arrays (keys starting with ``mask_``).
    tokenizer : PreTrainedTokenizer | PreTrainedTokenizerFast
        Tokenizer used to decode IDs.
    mask_keys : list[str] | None
        Explicit list of mask keys; auto-detected if None.
    max_rows : int | None
        Optional limit of token rows (useful for very long sequences).

    """
    mask_keys = _derive_mask_keys(
        features=features,
        explicit=mask_keys,
    )
    if not mask_keys:
        return "(No mask_* keys found.)"

    console = Console(
        width=180,
        record=True,
        soft_wrap=False,
    )

    for batch_idx in range(len(features.input_ids)):
        _render_single_example(
            console=console,
            tokenizer=tokenizer,
            features=features,
            batch_idx=batch_idx,
            mask_keys=mask_keys,
            max_rows=max_rows,
        )

    return console.export_text()


def _derive_mask_keys(
    features: BatchEncoding,
    explicit: list[str] | None,
) -> list[str]:
    if explicit is not None:
        return explicit
    return [k for k in features if k.startswith("mask_")]


def _clean_token(tok: str) -> str:
    if tok == "":
        return "∅"
    return tok.replace(
        "\n",
        "⏎",
    ).replace(
        "\t",
        "⇥",
    )


def _render_single_example(
    *,
    console: Console,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    features: BatchEncoding,
    batch_idx: int,
    mask_keys: list[str],
    max_rows: int | None,
) -> None:
    tokens: list[str] = tokenizer.convert_ids_to_tokens(
        features.input_ids[batch_idx],
    )  # type: ignore - we know the converted type is list[str]
    masks_per_key: dict[
        str,
        list[int],
    ] = {k: list(features[k][batch_idx]) for k in mask_keys if k in features}  # type: ignore[index]

    table = Table(
        title=f"Example {batch_idx} (tokens={len(tokens)})",
        show_header=True,
        header_style="bold magenta",
        box=box.SIMPLE_HEAVY,
        padding=(0, 1),
    )
    table.add_column(
        header="Idx",
        style="bold cyan",
        no_wrap=True,
    )
    table.add_column(
        header="Token",
        style="bold white",
    )
    for mk in mask_keys:
        if mk in masks_per_key:
            table.add_column(
                header=mk,
                style="bold green",
                no_wrap=True,
            )

    limit = len(tokens) if max_rows is None else min(max_rows, len(tokens))
    for i in range(limit):
        row: list[str] = [str(i), _clean_token(tokens[i])]
        for mk in mask_keys:
            if mk in masks_per_key:
                v = masks_per_key[mk][i]
                row.append("[bold green]1[/]" if v == 1 else "0")
        table.add_row(*row)
    if max_rows is not None and limit < len(tokens):
        ellipsis_row: list[str] = ["…", f"(truncated {len(tokens) - limit} tokens)"]
        ellipsis_row.extend(["…" for _ in range(len(table.columns) - 2)])
        table.add_row(*ellipsis_row)
    console.print(table)


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
