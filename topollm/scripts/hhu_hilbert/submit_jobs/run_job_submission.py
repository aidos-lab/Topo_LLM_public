import logging
import pathlib

from topollm.config_classes.submit_jobs.machine_configuration_config import (
    MachineConfigurationConfig,
    get_machine_configuration_args_list,
)
from topollm.scripts.hhu_hilbert.submit_jobs.call_command import call_command
from topollm.typing.enums import JobRunMode, Verbosity

default_logger = logging.getLogger(__name__)


def run_job_submission(
    python_script_absolute_path: pathlib.Path,
    job_script_args: list[str],
    machine_configuration: MachineConfigurationConfig,
    job_name: str = "default_wandb_project_name_0",
    verbosity: Verbosity = Verbosity.NORMAL,
    logger: logging.Logger = default_logger,
) -> None:
    """Run the job submission by assembling the command and calling it."""
    job_script_args_str = " ".join(
        job_script_args,
    )
    if verbosity >= Verbosity.NORMAL:
        logger.info(
            f"JOB_SCRIPT_ARGS={job_script_args_str}",  # noqa: G004 - low overhead
        )

    machine_configuration_args_list = get_machine_configuration_args_list(
        machine_configuration_config=machine_configuration,
    )

    if machine_configuration.job_run_mode == JobRunMode.HHU_HILBERT:
        command: list[str] = [
            *machine_configuration.submit_job_hilbert_command,
            "--job_name",
            str(job_name),
            "--job_script",
            str(python_script_absolute_path),
            *machine_configuration_args_list,
            "--job_script_args",
            job_script_args_str,
        ]
    elif machine_configuration.job_run_mode == JobRunMode.LOCAL:
        command = [
            *machine_configuration.run_job_locally_command,
            str(python_script_absolute_path),
            *job_script_args,
        ]
    else:
        msg = f"Invalid: {machine_configuration.job_run_mode = }"
        raise ValueError(msg)

    # Logging of the command is done in the `call_command` function
    call_command(
        command=command,
        dry_run=machine_configuration.dry_run,
        verbosity=verbosity,
        logger=logger,
    )
