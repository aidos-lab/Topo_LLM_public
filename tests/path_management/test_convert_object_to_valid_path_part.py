"""Tests for the functions which convert objects into valid path parts."""

import logging

from topollm.path_management.convert_object_to_valid_path_part import convert_list_to_path_part
from topollm.path_management.validate_path_part import validate_path_part


def test_convert_list_to_path_part(
    logger_fixture: logging.Logger,
) -> None:
    """Example usage of convert_str_enum_list_to_path_part function."""
    list_of_examples: list[list] = [
        [],  # empty list
        [
            "",
        ],  # list with one empty string
        [
            "encoder.layer.0.",
            "encoder.layer.1.",
            "encoder.layer.2.",
            "encoder.layer.3.",
            "encoder.layer.4.",
            "encoder.layer.5.",
        ],  # list with multiple strings
        [
            "query",
            "key",
            "value",
        ],  # list with multiple strings
        [
            -1,
            -2,
            -3,
        ],  # list with multiple integers
    ]

    for example in list_of_examples:
        path_part = convert_list_to_path_part(
            input_list=example,
        )
        logger_fixture.info(
            "path_part:\n%s",
            path_part,
        )

        assert validate_path_part(  # noqa: S101 - pytest assertion
            path_part=path_part,
        )
