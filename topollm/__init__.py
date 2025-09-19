"""Init file for the topollm package."""

try:
    from hydra_plugins import hpc_submission_launcher

    hpc_submission_launcher.register_plugin()
    print(  # noqa: T201 - We want this status to print
        "hpc_submission_launcher hydra plugin registered.",
    )
except ImportError:
    print(  # noqa: T201 - We want this status to print
        "WARNING: hpc_submission_launcher hydra plugin not found!",
    )
    pass  # noqa: PIE790 - pass statement to not have empty clause if print is removed
