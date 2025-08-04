"""Sanitize a directory name by replacing all slashes with underscores."""

from topollm.path_management.truncate_length_of_desc import truncate_length_of_desc


def sanitize_dirname(
    dir_name: str,
) -> str:
    """Sanitizes a directory name by replacing all slashes with underscores.

    Args:
    ----
        dir_name: The directory name to sanitize.

    Returns:
    -------
        The sanitized directory name.

    """
    result = dir_name.replace(
        "/",
        "_",
    ).replace(
        "\\",
        "_",
    )

    result: str = truncate_length_of_desc(
        desc=result,
    )

    if len(result) == 0:
        result = "no_overrides"

    return result
