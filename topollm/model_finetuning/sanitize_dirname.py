def sanitize_dirname(
    dir_name: str,
) -> str:
    """
    Sanitizes a directory name by replacing all slashes with underscores.

    Args:
        dir_name: The directory name to sanitize.

    Returns:
        The sanitized directory name.
    """
    result = dir_name.replace(
        "/",
        "_",
    ).replace(
        "\\",
        "_",
    )

    return result
