# Copyright 2024-2025
# [ANONYMIZED_INSTITUTION],
# [ANONYMIZED_FACULTY],
# [ANONYMIZED_DEPARTMENT]
#
# Authors:
# AUTHOR_1 (author1@example.com)
# AUTHOR_2 (author2@example.com)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#


"""Sort a list of strings representing integers in ascending order."""


def main() -> None:
    """Sort a list of strings representing integers in ascending order."""
    # List of strings representing integers
    checkpoint_no_list: list[str] = [
        "109707",
        "11252",
        "115333",
        "126585",
        "14065",
        "16878",
        "19691",
        "2813",
        "25317",
        "33756",
        "36569",
        "39382",
        "42195",
        "50634",
        "5626",
        "56260",
        "70325",
        "8439",
        "90016",
    ]

    # Sorting the list based on numerical values
    sorted_checkpoint_no_list = sorted(
        checkpoint_no_list,
        key=int,
    )

    # Output the sorted list
    print(  # noqa: T201 - we want this script to print the output
        sorted_checkpoint_no_list,
    )

    # > Expected output:
    # >
    # > ['2813', '5626', '8439', '11252', '14065', '16878', '19691', '25317', '33756', '36569', '39382', '42195', '50634', '56260', '70325', '90016', '109707', '115333', '126585']


if __name__ == "__main__":
    main()
