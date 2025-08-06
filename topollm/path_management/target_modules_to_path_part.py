from topollm.path_management.convert_object_to_valid_path_part import convert_list_to_path_part


def target_modules_to_path_part(
    target_modules: list[str] | str | None,
) -> str:
    """Convert the target_modules to a path part.

    Notes:
    - If target_modules is None, the function returns "None".
    - If target_modules is a string, it is returned as is.
    - For an empty list, the function returns "None" and not the empty string.

    """
    if target_modules is None:
        target_modules_path_part: str = "None"
    elif isinstance(
        target_modules,
        str,
    ):
        target_modules_path_part: str = target_modules
    elif isinstance(
        target_modules,
        list,
    ):
        if len(target_modules) == 0:
            target_modules_path_part: str = "None"
        else:
            target_modules_path_part: str = convert_list_to_path_part(
                input_list=target_modules,
            )
    else:
        msg: str = f"Unknown type of target_modules: {type(target_modules) = }. Cannot convert to path part."
        raise TypeError(
            msg,
        )

    return target_modules_path_part
