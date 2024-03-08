from topollm.storage.StorageProtocols import ArrayDataChunk, ArrayProperties, ChunkIdentifier, StoragePaths


import zarr


import logging
import os
import warnings


class ZarrChunkedArrayStorage:
    """
    A storage protocol backend for token level embeddings
    using Zarr and Xarray.

    Note: We do not need to inherit from TokenLevelEmbeddingStorageProtocol,
    since we are not relying on an abstract base class.
    """
    # TODO Implement this class

    def __init__(
        self,
        array_properties: ArrayProperties,
        storage_paths: StoragePaths,
        logger: logging.Logger = logging.getLogger(__name__),
    ):
        self.array_properties = array_properties
        self.storage_paths = storage_paths

    def open(
        self,
    ) -> None:
        # # # #
        # Open zarr array (for embeddings)
        os.makedirs(
            self.storage_paths.array_dir,
            exist_ok=True,
        )
        self.zarr_array = zarr.open(
            store=self.storage_paths.array_dir,  # type: ignore
            mode="w",
            shape=self.array_properties.shape,
            dtype=self.array_properties.dtype,
            chunks=self.array_properties.chunks,
        )

        # # # #
        # Open xarray (for metadata)

        warnings.warn(
            message=f"xarray Not implemented yet",
        )

        # TODO 2: Continue here

        return

    def write_chunk(
        self,
        data_chunk: ArrayDataChunk,
    ) -> None:
        # TODO: Update this to work with the DataClass

        # TODO 1: Implement saving of the embeddings
        # TODO 2: Implement saving of the metadata

        warnings.warn(
            message=f"write_chunk Not implemented yet",
        )

        return  # TODO fake implementation

        self.zarr_array[start_idx : start_idx + len(data)] = data

    def read_chunk(
        self,
        chunk_identifier: ChunkIdentifier,
    ) -> ArrayDataChunk:
        # TODO implement

        raise NotImplementedError