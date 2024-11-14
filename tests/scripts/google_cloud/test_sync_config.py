# Copyright 2024-2025
# Heinrich Heine University Dusseldorf,
# Faculty of Mathematics and Natural Sciences,
# Computer Science Department
#
# Authors:
# Benjamin Ruppik (mail@ruppik.net)
# Julius von Rohrscheidt (julius.rohrscheidt@helmholtz-muenchen.de)
#
# Code generation tools and workflows:
# First versions of this code were potentially generated
# with the help of AI writing assistants including
# GitHub Copilot, ChatGPT, Microsoft Copilot, Google Gemini.
# Afterwards, the generated segments were manually reviewed and edited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test the SyncConfig."""

import logging
import pprint

import pytest

from topollm.scripts.google_cloud.sync_config import SyncConfig


@pytest.fixture
def valid_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fixture to set all required environment variables."""
    monkeypatch.setenv(
        name="LOCAL_TOPO_LLM_DATA_DIR",
        value="/test/local/data/dir",
    )
    monkeypatch.setenv(
        name="GC_DEV_VM_HOSTNAME",
        value="test-vm-hostname",
    )
    monkeypatch.setenv(
        name="GC_DEV_VM_DATA_DIR",
        value="/test/vm/data/dir",
    )


@pytest.fixture
def missing_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fixture to simulate a missing environment variable."""
    # Set all required variables except one to test missing variable scenario
    monkeypatch.setenv(
        name="LOCAL_TOPO_LLM_DATA_DIR",
        value="/test/local/data/dir",
    )
    monkeypatch.setenv(
        name="GC_DEV_VM_HOSTNAME",
        value="test-vm-hostname",
    )


def test_sync_config_load_from_env_file(
    logger_fixture: logging.Logger,
) -> None:
    """Test loading the SyncConfig with all environment variables properly set."""
    config: SyncConfig = SyncConfig.load_from_env()

    logger_fixture.info(
        msg=f"config:\n{pprint.pformat(object=config)}",  # noqa: G004 - low overhead
    )


def test_sync_config_load_from_env_with_monkeypatch(
    valid_env_vars: None,  # noqa: ARG001 - fixture
    logger_fixture: logging.Logger,
) -> None:
    """Test loading the SyncConfig with all environment variables properly set."""
    config: SyncConfig = SyncConfig.load_from_env()

    logger_fixture.info(
        msg=f"config:\n{pprint.pformat(object=config)}",  # noqa: G004 - low overhead
    )

    assert config.local_data_dir == "/test/local/data/dir"  # noqa: S101 - pytest assertion
    assert config.gc_vm_hostname == "test-vm-hostname"  # noqa: S101 - pytest assertion
    assert config.gc_vm_data_dir == "/test/vm/data/dir"  # noqa: S101 - pytest assertion


def test_sync_config_with_explicit_overwrite(
    valid_env_vars: None,  # noqa: ARG001 - fixture
    logger_fixture: logging.Logger,
) -> None:
    """Test that the SyncConfig properly applies explicit overwrite for local_data_dir."""
    overwrite_value = "/test/overwrite/data/dir"

    config: SyncConfig = SyncConfig.load_from_env(
        local_data_dir_overwrite=overwrite_value,
    )

    logger_fixture.info(
        msg=f"config:\n{pprint.pformat(object=config)}",  # noqa: G004 - low overhead
    )

    assert config.local_data_dir == overwrite_value  # noqa: S101 - pytest assertion
    assert config.gc_vm_hostname == "test-vm-hostname"  # noqa: S101 - pytest assertion
    assert config.gc_vm_data_dir == "/test/vm/data/dir"  # noqa: S101 - pytest assertion
