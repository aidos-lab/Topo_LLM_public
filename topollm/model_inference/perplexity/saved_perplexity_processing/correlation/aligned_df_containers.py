"""Dataclasses to manage aligned_df.csv files and their metadata."""

import os
import pathlib
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class AlignedDFMetadata:
    """Dataclass that holds metadata extracted from the file path of an aligned_df.csv file."""

    dataset: str | None = None
    model: str | None = None
    checkpoint: str | None = None

    @staticmethod
    def from_path(
        file_path: os.PathLike,
    ) -> "AlignedDFMetadata":
        """Extract metadata from the given file path.

        Args:
        ----
            file_path (str): The full path to an aligned_df.csv file.

        Returns:
        -------
            AlignedDFMetadata: An instance of AlignedDFMetadata with extracted metadata.

        """
        file_path = pathlib.Path(file_path)

        components = file_path.parts
        dataset, model, checkpoint = None, None, None

        for component in components:
            if "data-" in component:
                dataset = component.split("data-")[-1]
            if "model-" in component:
                model = component.split("model-")[-1]
            if "ckpt-" in component:
                checkpoint = component.split("ckpt-")[-1].split("_")[0]

        return AlignedDFMetadata(dataset=dataset, model=model, checkpoint=checkpoint)

    def model_without_checkpoint(self) -> str | None:
        """Return the model identifier without the checkpoint, but includes any suffix after the checkpoint.

        Returns
        -------
            str | None: The model identifier without the checkpoint.

        """
        if self.model:
            parts = self.model.split("_ckpt-")
            if len(parts) > 1:
                # Reassemble the model string without the checkpoint but include the part after it
                return f"{parts[0]}_{parts[1].split('_', 1)[1]}"
            return parts[0]
        return None


@dataclass
class AlignedDF:
    """Dataclass that holds an aligned_df.csv file and its metadata.

    This includes:
    - path to an aligned_df.csv file,
    - the associated DataFrame,
    - and metadata extracted from the file path.
    """

    file_path: pathlib.Path
    dataframe: pd.DataFrame = field(
        init=False,
    )
    metadata: AlignedDFMetadata = field(
        init=False,
    )

    def __post_init__(self) -> None:
        """Initialize the DataFrame and metadata from the file path."""
        self.dataframe = pd.read_csv(
            self.file_path,
        )
        self.metadata = AlignedDFMetadata.from_path(
            self.file_path,
        )


@dataclass
class AlignedDFCollection:
    """Dataclass to manage a collection of AlignedDF objects for analysis."""

    aligned_dfs: list[AlignedDF] = field(
        default_factory=list,
    )

    def __len__(self) -> int:
        """Return the number of AlignedDF objects in the collection."""
        return len(self.aligned_dfs)

    def __getitem__(
        self,
        index: int,
    ) -> AlignedDF:
        """Return the AlignedDF object at the given index."""
        return self.aligned_dfs[index]

    def add_aligned_df(
        self,
        aligned_df: AlignedDF,
    ) -> None:
        self.aligned_dfs.append(aligned_df)

    def filter_by_model(
        self,
        model_name: str,
    ) -> list[AlignedDF]:
        return [df for df in self.aligned_dfs if df.metadata.model == model_name]

    def filter_by_dataset(
        self,
        dataset_name: str,
    ) -> list[AlignedDF]:
        return [df for df in self.aligned_dfs if df.metadata.dataset == dataset_name]

    def filter_by_dataset_and_model(
        self,
        dataset_name: str,
        model_without_ckpt: str,
    ) -> list[AlignedDF]:
        """Filter AlignedDF objects by dataset name and model identifier without checkpoint.

        Args:
        ----
            dataset_name (str): The name of the dataset.
            model_without_ckpt (str): The model identifier without the checkpoint.

        Returns:
        -------
            list[AlignedDF]: A list of filtered AlignedDF objects.

        """
        return [
            df
            for df in self.aligned_dfs
            if df.metadata.dataset == dataset_name and df.metadata.model_without_checkpoint() == model_without_ckpt
        ]

    def filter_by_checkpoint(
        self,
        checkpoint: str,
    ) -> list[AlignedDF]:
        return [df for df in self.aligned_dfs if df.metadata.checkpoint == checkpoint]

    def get_aggregated_statistics(
        self,
        statistic: str = "mean",
    ) -> pd.DataFrame:
        """Aggregate the statistics from all loaded DataFrames based on the selected statistic.

        Args:
        ----
            statistic:
                The statistic to aggregate (e.g., 'mean', 'std', 'min', 'max').

        Returns:
        -------
            pd.DataFrame: Aggregated statistics for the selected statistic, with metadata.

        """
        aggregated_data = []

        for df in self.aligned_dfs:
            stats: pd.DataFrame = df.dataframe.describe().transpose()
            if statistic not in stats.columns:
                msg = f"{statistic = } is not available in the DataFrame."
                raise ValueError(msg)

            # Create a new DataFrame to hold the selected statistic and metadata
            selected_stats = pd.DataFrame(
                {column: [stats.at[column, statistic]] for column in stats.index},  # noqa: PD008 - we want to access a single value
            )
            selected_stats["dataset"] = df.metadata.dataset
            selected_stats["model"] = df.metadata.model
            selected_stats["checkpoint"] = df.metadata.checkpoint
            selected_stats["model_without_checkpoint"] = df.metadata.model_without_checkpoint()
            selected_stats["count"] = stats.iloc[0][
                "count"
            ]  # Since the "count" value is the same for all columns, we can access it from the first row

            aggregated_data.append(selected_stats)

        # Concatenate all statistics into a single DataFrame
        result_df = pd.concat(
            aggregated_data,
            axis=0,
        ).reset_index(
            drop=True,
        )

        return result_df
