"""Utility functions for extracting spans from text and creating token-level masks."""

import re
from typing import Literal

from rich import box
from rich.console import Console
from rich.table import Table
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerFast


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
    last: tuple[int, int] | None = None

    for m in pattern.finditer(
        string=text,
    ):
        (
            marker_start,
            marker_end,
        ) = m.start(), m.end()
        seg_end: int = text.find(
            delimiter,
            marker_end,
        )
        if seg_end == -1:
            seg_end = len(text)
        span_start: int = marker_start if include_marker else marker_end
        span_end: int = seg_end
        last = (
            span_start,
            span_end,
        )

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
        row: list[str] = [
            str(i),
            _clean_token(tokens[i]),
        ]
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
