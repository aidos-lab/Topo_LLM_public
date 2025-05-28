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


from topollm.embeddings_data_prep.prepared_data_containers import PreparedData


def truncate_prepared_data(
    prepared_data: PreparedData,
    local_estimates_sample_size: int,
) -> PreparedData:
    """Truncate the data to the first `local_estimates_sample_size` samples."""
    input_array = prepared_data.array

    local_estimates_sample_size = min(
        local_estimates_sample_size,
        input_array.shape[0],
    )

    output_array = input_array[:local_estimates_sample_size,]

    input_meta_frame = prepared_data.meta_df
    output_meta_frame = input_meta_frame.iloc[:local_estimates_sample_size,]

    output_prepared_data = PreparedData(
        array=output_array,
        meta_df=output_meta_frame,
    )

    return output_prepared_data
