"""Tests for the embeddings path manager."""

import logging
import pathlib

from topollm.path_management.embeddings import protocol
from topollm.path_management.validate_path_part import validate_path_part


def validate_path(
    path: pathlib.Path,
) -> None:
    """Validate the path."""
    assert isinstance(  # noqa: S101 - pytest assertion
        path,
        pathlib.Path,
    )

    assert validate_path_part(  # noqa: S101 - pytest assertion
        path_part=str(object=path),
    )


class TestEmbeddingsPathManager:
    """Tests for the embeddings path manager."""

    def test_data_dir(
        self,
        embeddings_path_manager: protocol.EmbeddingsPathManager,
        logger_fixture: logging.Logger,
    ) -> None:
        result: pathlib.Path = embeddings_path_manager.data_dir
        logger_fixture.info(
            "data_dir:\n%s",
            result,
        )

        validate_path(
            path=result,
        )

    def test_array_dir_absolute_path(
        self,
        embeddings_path_manager: protocol.EmbeddingsPathManager,
        logger_fixture: logging.Logger,
    ) -> None:
        result: pathlib.Path = embeddings_path_manager.array_dir_absolute_path
        logger_fixture.info(
            "array_dir_absolute_path:\n%s",
            result,
        )

        validate_path(
            path=result,
        )

    def test_metadata_dir_absolute_path(
        self,
        embeddings_path_manager: protocol.EmbeddingsPathManager,
        logger_fixture: logging.Logger,
    ) -> None:
        result: pathlib.Path = embeddings_path_manager.metadata_dir_absolute_path
        logger_fixture.info(
            "metadata_dir_absolute_path:\n%s",
            result,
        )

        validate_path(
            path=result,
        )

    def test_get_global_estimate_save_path(
        self,
        embeddings_path_manager: protocol.EmbeddingsPathManager,
        logger_fixture: logging.Logger,
    ) -> None:
        result: pathlib.Path = embeddings_path_manager.get_global_estimate_save_path()
        logger_fixture.info(
            "global_estimate_save_path:\n%s",
            result,
        )

        validate_path(
            path=result,
        )

    def test_get_local_estimates_pointwise_array_save_path(
        self,
        embeddings_path_manager: protocol.EmbeddingsPathManager,
        logger_fixture: logging.Logger,
    ) -> None:
        result: pathlib.Path = embeddings_path_manager.get_local_estimates_pointwise_array_save_path()
        logger_fixture.info(
            "local_estimates_pointwise_array_save_path:\n%s",
            result,
        )

        validate_path(
            path=result,
        )

    def test_get_local_estimates_pointwise_meta_save_path(
        self,
        embeddings_path_manager: protocol.EmbeddingsPathManager,
        logger_fixture: logging.Logger,
    ) -> None:
        result: pathlib.Path = embeddings_path_manager.get_local_estimates_pointwise_meta_save_path()
        logger_fixture.info(
            "local_estimates_pointwise_meta_save_path:\n%s",
            result,
        )

        validate_path(
            path=result,
        )
