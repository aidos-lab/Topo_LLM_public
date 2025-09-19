"""Unit tests for span extraction and token mask creation functions."""

import pytest

from topollm.compute_embeddings.embedding_dataloader_preparer.extract_spans import _mask_from_span


def test_none_span_returns_all_zeros() -> None:
    """Span=None -> mask all zeros."""
    offsets: list[tuple[int, int]] = [(0, 3), (4, 7), (8, 12)]
    mask: list[int] = _mask_from_span(offsets=offsets, span=None, special_tokens_mask=None)
    assert mask == [0, 0, 0]  # noqa: S101 - pytest assert


def test_exact_match_span_sets_ones() -> None:
    """Span exactly equals one token's [start,end) -> that token is 1."""
    offsets: list[tuple[int, int]] = [(0, 3), (3, 10), (10, 20)]
    mask: list[int] = _mask_from_span(offsets=offsets, span=(3, 10), special_tokens_mask=None)
    assert mask == [0, 1, 0]  # noqa: S101 - pytest assert


def test_half_open_boundaries_behavior() -> None:
    """Half-open intervals: tokens ending at span.start or starting at span.end are 0."""
    offsets: list[tuple[int, int]] = [(5, 10), (10, 15), (15, 20), (20, 25)]
    mask: list[int] = _mask_from_span(offsets=offsets, span=(10, 20), special_tokens_mask=None)
    assert mask == [0, 1, 1, 0]  # noqa: S101 - pytest assert


def test_partial_overlap_sets_one() -> None:
    """Any non-empty overlap with span is 1."""
    offsets: list[tuple[int, int]] = [(0, 5), (9, 11), (19, 21)]
    mask: list[int] = _mask_from_span(offsets=offsets, span=(10, 20), special_tokens_mask=None)
    assert mask == [0, 1, 1]  # noqa: S101 - pytest assert


def test_zero_length_span() -> None:
    """Zero-width span [a,a)."""
    offsets: list[tuple[int, int]] = [(0, 5), (5, 10), (10, 15)]
    mask: list[int] = _mask_from_span(offsets=offsets, span=(7, 7), special_tokens_mask=None)
    assert mask == [0, 1, 0]  # noqa: S101 - pytest assert


def test_offsets_with_zero_zero_are_ignored() -> None:
    """Offsets (0,0) treated as special/pad -> remain 0 even if span overlaps elsewhere."""
    offsets: list[tuple[int, int]] = [(0, 0), (5, 10), (10, 15), (0, 0)]
    mask: list[int] = _mask_from_span(offsets=offsets, span=(6, 14), special_tokens_mask=None)
    assert mask == [0, 1, 1, 0]  # noqa: S101 - pytest assert


def test_special_tokens_mask_forces_zero() -> None:
    """special_tokens_mask=1 forces 0 even if the token overlaps span."""
    offsets: list[tuple[int, int]] = [(0, 2), (2, 6), (6, 9)]
    span: tuple[int, int] = (1, 8)  # would overlap tokens 0,1,2 but token 1 is forced to 0
    special_tokens_mask: list[int] = [0, 1, 0]
    mask: list[int] = _mask_from_span(offsets=offsets, span=span, special_tokens_mask=special_tokens_mask)
    assert mask == [1, 0, 1]  # noqa: S101 - pytest assert


@pytest.mark.parametrize(
    argnames=("offsets", "span", "expected"),
    argvalues=[
        # span fully before tokens -> all zeros
        ([(10, 20), (21, 30)], (0, 5), [0, 0]),
        # span fully after tokens -> all zeros
        ([(0, 5), (5, 9)], (10, 12), [0, 0]),
        # mixed overlaps -> only overlapping tokens are 1
        ([(0, 3), (3, 10), (11, 20)], (8, 12), [0, 1, 1]),
    ],
)
def test_various_overlap_patterns(offsets: list[tuple[int, int]], span: tuple[int, int], expected: list[int]) -> None:
    """Regression: cover before/after/mixed overlap patterns."""
    mask: list[int] = _mask_from_span(offsets=offsets, span=span, special_tokens_mask=None)
    assert mask == expected  # noqa: S101 - pytest assert
