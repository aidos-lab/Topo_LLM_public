"""Test the mask-based vector selection in a 3D array."""

import numpy as np


def test_mask_vector_selection() -> None:
    """Verify mask-based vector selection in a 3D array with detailed debugging output."""
    # Create a 3D array with clear, incremental values
    # Dimensions: `(batch, sequence, embedding)`
    embeddings = np.zeros(
        (5, 7, 3),
        dtype=float,
    )

    # Fill with incremental values to make verification easy
    for batch in range(5):
        for seq in range(7):
            for emb in range(3):
                embeddings[batch, seq, emb] = batch * 100 + seq * 10 + emb

    # Print the entire embedding array for visual inspection
    print("Full Embedding Array:")
    print(embeddings)
    print("\nArray Shape:", embeddings.shape)

    # Indices we want to test (matching your original description)
    mask_indices = (
        np.array([0, 1, 2, 3, 4]),
        np.array([1, 2, 3, 4, 5]),
    )

    # Print mask indices for clarity
    print("\nMask Indices:")
    print("Batch indices:", mask_indices[0])
    print("Sequence indices:", mask_indices[1])

    # Perform the selection.
    # Equivalent line:
    # > selected_vectors = embeddings[mask_indices[0], mask_indices[1], :]
    selected_vectors = embeddings[mask_indices]

    # Print selected vectors for visual inspection
    print("\nSelected Vectors:")
    print(selected_vectors)
    print("Selected Vectors Shape:", selected_vectors.shape)

    # Manually verify each selected vector
    print("\nDetailed Vector Verification:")
    for i, (batch, seq) in enumerate(
        iterable=zip(
            mask_indices[0],
            mask_indices[1],
            strict=True,
        ),
    ):
        print(f"Vector {i} - Batch {batch}, Sequence {seq}:")
        print("  Original vector:", embeddings[batch, seq, :])
        print("  Extracted vector:", selected_vectors[i])

        # Optional: Assert they are exactly the same
        assert np.array_equal(  # noqa: S101 - pytest assert
            embeddings[batch, seq, :],
            selected_vectors[i],
        ), f"Mismatch at batch {batch}, sequence {seq}"

    print("\nAll vector selections verified successfully!")


# Run the test
if __name__ == "__main__":
    test_mask_vector_selection()
