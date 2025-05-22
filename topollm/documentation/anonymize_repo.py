"""Anonymize sensitive information in a repository."""

import json
import pathlib
import pprint
import re


def load_replacements(
    mapping_path: pathlib.Path,
) -> dict[str, str]:
    """Load anonymization replacements from a JSON file."""
    with mapping_path.open(
        mode="r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def anonymize_file(
    filepath: pathlib.Path,
    replacements: dict[str, str],
) -> None:
    """Replace sensitive info in a file with anonymized placeholders."""
    try:
        text = filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Skip binary or non-UTF-8 files
        return
    original_text = text
    for pattern, replacement in replacements.items():
        text = re.sub(re.escape(pattern), replacement, text)
    if text != original_text:
        filepath.write_text(text, encoding="utf-8")


def process_repo(
    root: pathlib.Path,
    replacements: dict[str, str],
    exclude_dirs: set[str] | None = None,
    exclude_files: set[str] | None = None,
) -> None:
    """Recursively anonymize files in a repository."""
    if exclude_dirs is None:
        exclude_dirs = {
            ".git",
            ".venv",
            "data",
            "hydra_output_dir",
            "hydra_plugins",
            "logs",
            "temp",
            "tests/temp_files",
            "tmp_trainer",
            "outputs",
            "wandb",
            "wandb_output_dir",
        }

    if exclude_files is None:
        # Note: Hidden files (starting with a dot) are excluded by default,
        # but it does not hurt to be explicit.
        exclude_files = {
            "anonymization_map.json",
            ".gitignore",
            ".env",
        }

    for filepath in root.rglob("*"):
        # Skip directories and certain excluded folders
        if filepath.is_dir():
            continue
        # Skip files in excluded directories or hidden files/folders
        if any(part.startswith(".") for part in filepath.parts):
            print(  # noqa: T201 - we want this script to print
                f"Skipping hidden file: {filepath}",
            )
            continue
        if any(part in exclude_dirs for part in filepath.parts):
            print(  # noqa: T201 - we want this script to print
                f"Skipping excluded directory: {filepath}",
            )
            continue
        if filepath.name in exclude_files:
            print(  # noqa: T201 - we want this script to print
                f"Skipping excluded file: {filepath}",
            )
            continue

        anonymize_file(
            filepath=filepath,
            replacements=replacements,
        )


def main() -> None:
    """Anonymize a repository."""
    repo_root: pathlib.Path = pathlib.Path(__file__).parent.parent.parent
    mapping_path: pathlib.Path = repo_root / "grokking" / "documentation" / "anonymization_map.json"

    print(  # noqa: T201 - we want this script to print
        f"Anonymizing repository at {repo_root=} using mapping file {mapping_path=}",
    )

    replacements: dict[str, str] = load_replacements(
        mapping_path=mapping_path,
    )
    print(  # noqa: T201 - we want this script to print
        f"Loaded replacements:{pprint.pformat(replacements)}",
    )

    process_repo(
        root=repo_root,
        replacements=replacements,
        exclude_dirs=None,
        exclude_files=None,
    )

    print(  # noqa: T201 - we want this script to print
        "Anonymization complete.",
    )


if __name__ == "__main__":
    main()
