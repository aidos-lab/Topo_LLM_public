# Copyright 2024
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


"""
Read ICLR text data from 'ICLR_Mistral_Embeddings.csv' and split to train/test/validation
"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Imports

# Standard library imports
import logging
import pathlib

# Third party imports
import pandas as pd
from sklearn.model_selection import train_test_split

# Local imports
from topollm.logging.setup_exception_logging import setup_exception_logging

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# START Globals

# A logger for this file
global_logger = logging.getLogger(__name__)

setup_exception_logging(
    logger=global_logger,
)

# torch.set_num_threads(1)

# END Globals
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def main():
    data_root_dir = pathlib.Path(
        pathlib.Path.home(),
        "git-source",
        "Topo_LLM",
        "data",
    )

    print(f"data_root_dir: {data_root_dir}")

    current_dataset_dir = pathlib.Path(
        data_root_dir,
        "datasets",
        "iclr_2024_submissions",
    )

    path_to_csv = pathlib.Path(
        current_dataset_dir,
        "ICLR_Mistral_Embeddings.csv",
    )

    df = pd.read_csv(
        path_to_csv,
    )
    df = df.iloc[:, :5]

    df["text"] = df["title"] + ". " + df["abstract"]
    df = df.loc[:, ["title", "abstract", "text"]]

    train, test = train_test_split(
        df,
        test_size=0.2,
    )
    test, validation = train_test_split(
        test,
        test_size=0.5,
    )

    train.to_csv(
        pathlib.Path(
            current_dataset_dir,
            "ICLR_train.csv",
        ),
        index=False,
    )
    test.to_csv(
        pathlib.Path(
            current_dataset_dir,
            "ICLR_test.csv",
        ),
        index=False,
    )
    validation.to_csv(
        pathlib.Path(
            current_dataset_dir,
            "ICLR_validation.csv",
        ),
        index=False,
    )


if __name__ == "__main__":
    main()

    print("Done")
