import numpy as np

from topollm.embeddings_data_prep.prepared_data_containers import PreparedData


class RemoveZeroVectorsFilter:
    """Filter to remove zero vectors from the input data."""

    def filter_data(
        self,
        prepared_data: PreparedData,
    ) -> PreparedData:
        """Filter the data for local estimates computation."""
        input_array = prepared_data.array

        indices_to_keep = ~np.all(
            input_array == 0,
            axis=1,
        )

        # Remove zero rows from the array
        output_array = input_array[indices_to_keep]

        # Take the same rows from the meta frame
        input_meta_frame = prepared_data.meta_df

        output_meta_frame = input_meta_frame.iloc[indices_to_keep]

        output_data = PreparedData(
            array=output_array,
            meta_df=output_meta_frame,
        )

        return output_data
