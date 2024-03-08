# read ICLR text data from 'ICLR_Mistral_Embeddings.csv' and split to train/test/validation

import pathlib

import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd

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

train, test = train_test_split(df, test_size=0.2)
test, validation = train_test_split(test, test_size=0.5)

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

pass
