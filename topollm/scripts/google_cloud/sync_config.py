"""Configuration for synchronization including VM hostname and base data directory."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Configuration dictionary to make variable names configurable
ENV_VARIABLE_NAMES: dict[str, str] = {
    "local_repository_base_path": "TOPO_LLM_REPOSITORY_BASE_PATH",
    "local_data_dir": "LOCAL_TOPO_LLM_DATA_DIR",
    "gc_vm_hostname": "GC_DEV_VM_HOSTNAME",
    "gc_vm_repository_base_path": "GC_DEV_VM_REPOSITORY_BASE_PATH",
    "gc_vm_data_dir": "GC_DEV_VM_DATA_DIR",
    "gc_bucket_name": "GC_TOPO_LLM_BUCKET_NAME",
    "gc_bucket_path": "GC_TOPO_LLM_BUCKET_PATH",
    "gc_bucket_repository_base_path": "GC_TOPO_LLM_BUCKET_REPOSITORY_BASE_PATH",
    "gc_bucket_data_dir": "GC_TOPO_LLM_BUCKET_DATA_DIR",
}


def get_env_variable(
    key: str,
    default: str | None = None,
) -> str:
    """Fetch environment variables and raise Error if None."""
    value: str | None = os.getenv(
        key=key,
        default=default,
    )
    if value is None:
        msg: str = f"Environment variable '{key = }' is not set."
        raise ValueError(msg)
    return value


@dataclass
class SyncConfig:
    """Configuration for synchronization including VM hostname and base data directory."""

    local_repository_base_path: str
    local_data_dir: str
    gc_vm_hostname: str
    gc_vm_data_dir: str
    gc_vm_repository_base_path: str
    gc_bucket_name: str
    gc_bucket_path: str
    gc_bucket_repository_base_path: str
    gc_bucket_data_dir: str

    @staticmethod
    def load_from_env(
        config_keys: dict[str, str] = ENV_VARIABLE_NAMES,
        local_data_dir_overwrite: str | None = None,
    ) -> "SyncConfig":
        """Load environment variables and initialize SyncConfig.

        Returns:
            SyncConfig: The synchronization configuration instance.

        """
        load_dotenv()

        # Local machine configuration
        local_repository_base_path: str = get_env_variable(
            key=config_keys["local_repository_base_path"],
        )
        local_data_dir: str = local_data_dir_overwrite or get_env_variable(
            key=config_keys["local_data_dir"],
        )

        # Google Cloud VM configuration
        gc_vm_hostname: str = get_env_variable(
            key=config_keys["gc_vm_hostname"],
        )
        gc_vm_repository_base_path: str = get_env_variable(
            key=config_keys["gc_vm_repository_base_path"],
        )
        gc_vm_data_dir: str = get_env_variable(
            key=config_keys["gc_vm_data_dir"],
        )

        # Google Cloud Bucket configuration
        gc_bucket_name: str = get_env_variable(
            key=config_keys["gc_bucket_name"],
        )
        gc_bucket_path: str = get_env_variable(
            key=config_keys["gc_bucket_path"],
        )
        gc_bucket_repository_base_path: str = get_env_variable(
            key=config_keys["gc_bucket_repository_base_path"],
        )
        gc_bucket_data_dir: str = get_env_variable(
            key=config_keys["gc_bucket_data_dir"],
        )

        sync_config = SyncConfig(
            local_repository_base_path=local_repository_base_path,
            local_data_dir=local_data_dir,
            gc_vm_hostname=gc_vm_hostname,
            gc_vm_repository_base_path=gc_vm_repository_base_path,
            gc_vm_data_dir=gc_vm_data_dir,
            gc_bucket_name=gc_bucket_name,
            gc_bucket_path=gc_bucket_path,
            gc_bucket_repository_base_path=gc_bucket_repository_base_path,
            gc_bucket_data_dir=gc_bucket_data_dir,
        )

        return sync_config
