"""Dataset entry -> feature conversion utilities and mask visualization.

Provides conversion helpers for different dataset types.
"""

import re
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


# List of known slot names used for parsing dialogue acts.
# This list is used to identify and separate slot names from their values when processing dialogue acts.
DEFAULT_DIALOGUE_ACT_SLOTS: list[str] = [
    "price range",
    "pricerange",
    "area",
    "food",
    "type",
    "name",
    "stars",
    "choice",
    "book time",
    "book day",
    "book people",
    "internet",
    "parking",
    "address",
    "entrance fee",
    "postcode",
    "phone",
    "ref",
    "reference",
]


def _extract_action_values(
    dialogue_act_text: str,
    slots: list[str],
) -> list[str]:
    """Return ordered list of candidate VALUE strings parsed from a dialogue act."""
    clauses: list[str] = [c.strip() for c in dialogue_act_text.split(";") if c.strip()]
    values: list[str] = []
    slots_sorted: list[str] = sorted(
        slots,
        key=len,
        reverse=True,
    )  # prefer multiword slots first

    for clause in clauses:
        parts: list[str] = clause.split()
        if not parts:
            continue
        # strip common intent token (inform / nooffer / request / recommend / etc.)
        intent: str = parts[0]
        tail: str = clause[len(intent) :].strip()

        # Try to match "<slot> <value>" (or "<slot>: <value>") with longest slot name
        matched_slot = None
        for sname in slots_sorted:
            low_tail = tail.lower()
            low_slot = sname.lower()
            if low_tail.startswith(low_slot + " ") or low_tail.startswith(low_slot + ":"):
                matched_slot = sname
                # everything after the slot name (minus separators) is the value
                raw_val = tail[len(sname) :].lstrip(" :")
                if raw_val:
                    values.append(raw_val)
                break
            if low_tail == low_slot:
                matched_slot = sname  # no value present (e.g., 'request area')
                break
        if matched_slot is not None:
            continue

        # Fallback: common numeric value pattern like "choice 18"
        m_choice = re.search(r"\bchoice\s+([0-9]+)\b", clause, flags=re.IGNORECASE)
        if m_choice:
            values.append(m_choice.group(1))
            continue

        # Last fallback: take the trailing token-ish phrase as a value guess
        m_tail = re.search(r"([A-Za-z0-9][A-Za-z0-9\s\-]*)$", tail)
        if m_tail:
            cand = m_tail.group(1).strip()
            if cand:
                values.append(cand)

    # Deduplicate but keep order
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        k = v.lower()
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out


def _compile_value_regex(value: str) -> re.Pattern[str]:
    """Case-insensitive whole-word-ish match; allow simple plurals for alpha values."""
    v = value.strip()
    if not v:
        return re.compile(r"(?!x)x")  # match nothing
    if v.isalpha():
        return re.compile(rf"\b{re.escape(v)}(?:s|es)?\b", flags=re.IGNORECASE)
    return re.compile(rf"\b{re.escape(v)}\b", flags=re.IGNORECASE)


def build_system_utterance_content_masks_from_dialogue_act_for_encoded_text(
    texts: list[str],
    encodings: BatchEncoding,
    *,
    delimiter: str = "</s>",
    dialogue_act_slots: list[str] = DEFAULT_DIALOGUE_ACT_SLOTS,
) -> dict[str, list[list[int]]]:
    """Create masks over the last system utterance: content (union of action values) and non-content."""
    offsets_batch: list[list[tuple[int, int]]] = encodings.offset_mapping
    specials_batch: list[list[int]] = encodings.special_tokens_mask

    out: dict[str, list[list[int]]] = {
        "mask_system_content": [],
        "mask_system_noncontent": [],
    }

    for text, offsets, specials in zip(
        texts,
        offsets_batch,
        specials_batch,
        strict=True,
    ):
        sys_span: tuple[int, int] | None = _find_last_label_span(
            text=text,
            label="system",
            delimiter=delimiter,
            include_marker=False,
        )
        act_span: tuple[int, int] | None = _find_last_label_span(
            text=text,
            label="action",
            delimiter=delimiter,
            include_marker=False,
        )

        # Base system span mask (all tokens of the last system utterance)
        system_span_mask = _mask_from_span(
            offsets=offsets,
            span=sys_span,
            special_tokens_mask=specials,
        )

        # If either span missing → empty content/non-content (or non-content = system, if you prefer)
        if sys_span is None or act_span is None:
            out["mask_system_content"].append([0] * len(offsets))
            out["mask_system_noncontent"].append([0] * len(offsets))
            continue

        # Extract action values and match them inside the last system span
        action_text: str = text[act_span[0] : act_span[1]]
        values: list[str] = _extract_action_values(
            dialogue_act_text=action_text,
            slots=dialogue_act_slots,
        )

        sys_sub: str = text[sys_span[0] : sys_span[1]]
        content_union: list[int] = [0] * len(offsets)

        for val in values:
            pat: re.Pattern[str] = _compile_value_regex(value=val)
            for m in pat.finditer(sys_sub):
                a: int = sys_span[0] + m.start()
                b: int = sys_span[0] + m.end()
                msk: list[int] = _mask_from_span(
                    offsets=offsets,
                    span=(a, b),
                    special_tokens_mask=specials,
                )
                content_union = [
                    int(x or y)
                    for x, y in zip(
                        content_union,
                        msk,
                        strict=True,
                    )
                ]

        # Non-content = tokens in system span AND not in content
        noncontent: list[int] = [
            int(s and not c)
            for s, c in zip(
                system_span_mask,
                content_union,
                strict=True,
            )
        ]

        out["mask_system_content"].append(content_union)
        out["mask_system_noncontent"].append(noncontent)

    return out


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
        "mask_state": [],
        "mask_database": [],
        "mask_action": [],
        "mask_system_last": [],
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
        span_sys: tuple[int, int] | None = _find_last_label_span(
            text=text,
            label="system",
            delimiter=delimiter,
            include_marker=False,
        )

        out["mask_state"].append(_mask_from_span(offsets=offsets, span=span_state, special_tokens_mask=specials))
        out["mask_database"].append(_mask_from_span(offsets=offsets, span=span_db, special_tokens_mask=specials))
        out["mask_action"].append(_mask_from_span(offsets=offsets, span=span_action, special_tokens_mask=specials))
        out["mask_system_last"].append(_mask_from_span(offsets=offsets, span=span_sys, special_tokens_mask=specials))

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

    system_content_masks: dict[str, list[list[int]]] = (
        build_system_utterance_content_masks_from_dialogue_act_for_encoded_text(
            texts=dataset_entry[column_name],
            encodings=tokenized_entries,
            delimiter="</s>",
            dialogue_act_slots=DEFAULT_DIALOGUE_ACT_SLOTS,
        )
    )

    # Combine the tokenized entries with the segment masks
    features = BatchEncoding(
        data={
            **tokenized_entries,
            **segment_masks,
            **system_content_masks,  # adds 'mask_system_content' and 'mask_system_noncontent'
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
