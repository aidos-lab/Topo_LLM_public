def validate_path_part(
    path_part: str,
) -> bool:
    """Validate if a string is suitable for file paths.

    This also checks for common issues that would appear when using gsutil for Google Cloud bucket operations,
    in particular, the following characters are not allowed because they might be interpreted as wildcard:
    `*`, `?`, `[`, `]`.
    Also see the following discussions:
    - https://stackoverflow.com/questions/42087510/gsutil-ls-returns-error-contains-wildcard
    - https://github.com/GoogleCloudPlatform/gsutil/issues/290
    - https://cloud.google.com/storage/docs/gsutil/addlhelp/WildcardNames
    """
    # Check if the path part is a string
    if not isinstance(
        path_part,
        str,
    ):
        return False

    gsutil_wildcard_characters: list[str] = [
        "*",
        "?",
        "[",
        "]",
    ]

    osx_forbidden_characters: list[str] = [
        ":",
    ]

    windows_forbidden_characters: list[str] = [
        "<",
        ">",
        ":",
        '"',
        "|",
        "?",
        "*",
    ]

    characters_to_avoid: list[str] = (
        gsutil_wildcard_characters + osx_forbidden_characters + windows_forbidden_characters
    )

    # Check if the path part does not contain any of the following characters
    if any(char in path_part for char in characters_to_avoid):
        return False
    return True
