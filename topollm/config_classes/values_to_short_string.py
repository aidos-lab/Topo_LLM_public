"""Convert various values to their short string representation."""

from topollm.typing.enums import DataSamplingMode, Split


def bool_to_short_string(
    *,
    value: bool,
) -> str:
    """Convert a boolean value to a short string representation."""
    return "T" if value else "F"


def split_to_short_string(
    split: Split,
) -> str:
    """Convert a Split enum to a short string representation."""
    match split:
        case Split.TRAIN:
            return "tr"
        case Split.VALIDATION | Split.DEV:
            return "va"
        case Split.TEST:
            return "te"
        case Split.FULL:
            return "fu"
        case _:
            msg: str = f"Invalid split: {split}"
            raise ValueError(msg)


def data_sampling_mode_to_short_string(
    data_sampling_mode: DataSamplingMode,
) -> str:
    """Convert a data sampling mode to a short string representation."""
    match data_sampling_mode:
        case DataSamplingMode.RANDOM:
            return "r"
        case DataSamplingMode.TAKE_FIRST:
            return "tf"
        case _:
            msg: str = f"Invalid {data_sampling_mode = }"
            raise ValueError(msg)
