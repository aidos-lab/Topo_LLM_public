"""Generate violin plots for local estimates from different models."""

import itertools
import logging
import pathlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum, auto, unique

import click
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from topollm.config_classes.constants import TOPO_LLM_REPOSITORY_BASE_PATH
from topollm.typing.enums import Verbosity

# ----------------------------- Enums & simple types -----------------------------

default_logger: logging.Logger = logging.getLogger(name=__name__)


@unique
class BaseModelMode(StrEnum):
    """The different modes for base models."""

    ROBERTA_BASE = auto()
    GPT2_MEDIUM = auto()

    LLAMA_32_1B = auto()
    LLAMA_32_3B = auto()
    LLAMA_31_8B = auto()

    PHI_35_MINI_INSTRUCT = auto()
    PHI_35_MINI_INSTRUCT_FOR_LUSTER_MODELS = auto()


@unique
class SaveFormat(StrEnum):
    """The different formats for saving plots."""

    PDF = "pdf"
    PNG = "png"


# --------------------------------- Dataclasses ---------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """A model (path segment used in data folder) with a human-readable label.

    Attributes:
        path: Exact directory segment used on disk, e.g. "model=roberta-base_task=...".
        label: Human-readable label for plots/legends.

    """

    path: str
    label: str


@dataclass(frozen=True)
class PlotConfig:
    """Plot-related configuration.

    Attributes:
        figsize: Matplotlib figure size.
        fontsize: Base font size for axes ticks/labels.
        legend_fontsize: Font size for legend.
        formats: Output formats to save.

    """

    figsize: tuple[float, float] = (7.0, 2.5)
    fontsize: int = 18
    legend_fontsize: int = 12
    formats: tuple[SaveFormat, ...] = (SaveFormat.PDF, SaveFormat.PNG)


@dataclass(frozen=True)
class RunConfig:
    """Global run configuration for the pipeline.

    Attributes:
        model_group: CLI-selected group filter.
        verbosity: Verbosity level.
        ddof: Delta degrees of freedom for std calculation.
        save_arrays_in_output_dir: Whether to copy arrays next to plots.
        add_prefix_space: Tokenization flag mirrored into path layout.
        data_subsampling_sampling_seed: Seed for dataset subsampling path segment.
        repo_base: Repository base path.
        dataset_name_choices: Datasets to iterate.
        split_choices: Splits to iterate.
        samples_choices: Sample counts to iterate.
        sampling_mode_choices: Sampling modes to iterate.
        edh_mode_choices: edh modes to iterate.
        plot: Plot configuration.

    """

    model_group: str
    verbosity: Verbosity = Verbosity.NORMAL
    ddof: int = 1
    save_arrays_in_output_dir: bool = True
    add_prefix_space: bool = False
    data_subsampling_sampling_seed: int = 778
    repo_base: pathlib.Path = pathlib.Path(TOPO_LLM_REPOSITORY_BASE_PATH)

    dataset_name_choices: list[str] = field(
        default_factory=lambda: [
            "data=iclr_2024_submissions_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags",
            "data=multiwoz21_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags",
            "data=one-year-of-tsla-on-reddit_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags",
            "data=sgd_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags",
            "data=wikitext-103-v1_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags",
            "data=wikitext-103-v1_strip-True_rm-empty=True_spl-mode=proportions_spl-shuf=True_spl-seed=0_tr=0.8_va=0.1_te=0.1_ctxt=dataset_entry_feat-col=ner_tags",
            # LUSTER data
            "data=luster_column=source_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags",
            "data=luster_column=source_target_rm-empty=True_spl-mode=do_nothing_ctxt=dataset_entry_feat-col=ner_tags",
        ],
    )
    split_choices: list[str] = field(default_factory=lambda: ["train", "validation", "test"])
    samples_choices: list[int] = field(default_factory=lambda: [7000, 10000])
    sampling_mode_choices: list[str] = field(default_factory=lambda: ["random", "take_first"])
    edh_mode_choices: list[str] = field(default_factory=lambda: ["masked_token", "regular"])

    plot: PlotConfig = field(default_factory=PlotConfig)

    @property
    def local_estimates_dir(self) -> pathlib.Path:
        """Root directory for local estimate arrays."""
        return self.repo_base / "data" / "analysis" / "local_estimates"

    @property
    def saved_plots_root(self) -> pathlib.Path:
        """Root directory for saved violin plots."""
        return self.repo_base / "data" / "saved_plots" / "violin_plots"


@dataclass(frozen=True)
class Task:
    """A single plotting task comparing two models on a dataset configuration.

    Attributes:
        dataset_name: Dataset directory segment (exact).
        split: Split name.
        samples: Number of samples.
        sampling_mode: Sampling mode ("random" | "take_first").
        edh_mode: edh mode ("masked_token" | "regular").
        first_model: Baseline model.
        second_model: Fine-tuned / comparison model.

    """

    dataset_name: str
    split: str
    samples: int
    sampling_mode: str
    edh_mode: str
    first_model: ModelSpec
    second_model: ModelSpec


# -------------------------------- Path Builder ---------------------------------


class PathBuilder:
    """Centralizes the directory conventions used in the repository."""

    def __init__(self, cfg: RunConfig) -> None:
        """Initialize a PathBuilder.

        Args:
            cfg: Global run configuration.

        """
        self.cfg = cfg

    def split_and_subsample_path(
        self,
        split: str,
        samples: int,
        sampling_mode: str,
    ) -> str:
        """Build the 'split=..._samples=..._sampling=...' segment.

        Args:
            split: Dataset split.
            samples: Sample count.
            sampling_mode: "random" or "take_first".

        Returns:
            Path segment string.

        Raises:
            ValueError: If sampling_mode is unsupported.

        """
        if sampling_mode == "random":
            return (
                f"split={split}"
                f"_samples={samples}"
                f"_sampling={sampling_mode}"
                f"_sampling-seed={self.cfg.data_subsampling_sampling_seed}"
            )
        if sampling_mode == "take_first":
            return f"split={split}_samples={samples}_sampling={sampling_mode}"
        msg = f"Unsupported sampling mode: {sampling_mode=}"
        raise ValueError(msg)

    def model_base_dir(
        self,
        dataset_name: str,
        split_and_subsample: str,
        edh_mode: str,
        model_path: str,
    ) -> pathlib.Path:
        """Build the base directory where the npy file lives for a given model.

        Args:
            dataset_name: Dataset segment.
            split_and_subsample: Output of split_and_subsample_path().
            edh_mode: EDH mode.
            model_path: Exact model directory segment.

        Returns:
            Absolute path to the model's base directory for arrays.

        """
        return (
            self.cfg.local_estimates_dir
            / dataset_name
            / split_and_subsample
            / f"edh-mode={edh_mode}_lvl=token"
            / f"add-prefix-space={self.cfg.add_prefix_space}_max-len=512"
            / model_path
            / "layer=-1_agg=mean"
            / "norm=None"
            / "sampling=random_seed=42_samples=150000"
            / "desc=twonn_samples=60000_zerovec=keep_dedup=array_deduplicator_noise=do_nothing"
            / "n-neighbors-mode=absolute_size_n-neighbors=128"
        )

    @staticmethod
    def estimates_file(base_dir: pathlib.Path) -> pathlib.Path:
        """Return the file path to the pointwise array."""
        return base_dir / "local_estimates_pointwise_array.npy"

    def plot_save_dir(
        self,
        second_model: ModelSpec,
        dataset_name: str,
        edh_mode: str,
        split_and_subsample: str,
    ) -> pathlib.Path:
        """Directory to save plots (and optionally arrays), path-stable.

        Uses EXACT `second_model.path` and `dataset_name` segments.

        Args:
            second_model: Comparison model spec.
            dataset_name: Dataset segment.
            edh_mode: EDH mode.
            split_and_subsample: Split/subsample segment.

        Returns:
            Absolute directory path to save outputs.

        """
        return (
            self.cfg.saved_plots_root
            / f"ddof={self.cfg.ddof}"
            / f"add-prefix-space={self.cfg.add_prefix_space}"
            / f"edh-mode={edh_mode}"
            / split_and_subsample
            / second_model.path  # exact model segment
            / dataset_name  # exact dataset segment
        )


# --------------------------- Model-spec registry -------------------------------


def model_specs_for(base_model_mode: BaseModelMode) -> tuple[ModelSpec, list[ModelSpec]]:
    """Return (first_model, list of second models) for a given base-model mode.

    Args:
        base_model_mode: Model family selector.

    Returns:
        Tuple of baseline ModelSpec and list of comparison ModelSpecs.

    Raises:
        ValueError: If base_model_mode is unsupported.

    """
    if base_model_mode is BaseModelMode.ROBERTA_BASE:
        ck = 2800
        first = ModelSpec("model=roberta-base_task=masked_lm_dr=defaults", "RoBERTa")
        seconds = [
            ModelSpec(
                f"model=roberta-base-masked_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-{ck}_task=masked_lm_dr=defaults",
                f"RoBERTa fine-tuned on MultiWOZ gs={ck}",
            ),
            ModelSpec(
                f"model=roberta-base-masked_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-{ck}_task=masked_lm_dr=defaults",
                f"RoBERTa fine-tuned on Reddit gs={ck}",
            ),
            ModelSpec(
                f"model=roberta-base-masked_lm-defaults_wikitext-103-v1-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-{ck}_task=masked_lm_dr=defaults",
                f"RoBERTa fine-tuned on Wikitext gs={ck}",
            ),
        ]
        return first, seconds

    if base_model_mode is BaseModelMode.GPT2_MEDIUM:
        ck = 1200
        first = ModelSpec("model=gpt2-medium_task=masked_lm_dr=defaults", "GPT-2")
        seconds = [
            ModelSpec(
                f"model=gpt2-medium-causal_lm-defaults_multiwoz21-rm-empty-True-do_nothing-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-{ck}_task=causal_lm_dr=defaults",
                f"GPT-2 fine-tuned on MultiWOZ gs={ck}",
            ),
            ModelSpec(
                f"model=gpt2-medium-causal_lm-defaults_one-year-of-tsla-on-reddit-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-{ck}_task=causal_lm_dr=defaults",
                f"GPT-2 fine-tuned on Reddit gs={ck}",
            ),
            ModelSpec(
                f"model=gpt2-medium-causal_lm-defaults_wikitext-103-v1-rm-empty-True-proportions-True-0-0.8-0.1-0.1-ner_tags_train-10000-take_first-111_standard-None_5e-05-linear-0.01-5_seed-1234_ckpt-{ck}_task=causal_lm_dr=defaults",
                f"GPT-2 fine-tuned on Wikitext gs={ck}",
            ),
        ]
        return first, seconds

    if base_model_mode is BaseModelMode.PHI_35_MINI_INSTRUCT:
        first = ModelSpec(path="model=Phi-3.5-mini-instruct_task=masked_lm_dr=defaults", label="Phi-3.5-mini-instruct")
        seconds: list[ModelSpec] = []
        for ck in [800, 1200, 2800]:
            seconds.extend(
                [
                    ModelSpec(
                        path="model=Phi-3.5-mini-instruct_multiwoz21_train-10000-r-778_aps-F-mx-512_lora-16-32-o_proj_qkv_proj-0.01-True_5e-05-linear-0.01-5"
                        f"_seed-1234_ckpt-{ck}_task=causal_lm_dr=defaults",
                        label=f"Phi-3.5-mini-instruct fine-tuned on MultiWOZ gs={ck}",
                    ),
                    ModelSpec(
                        path="model=Phi-3.5-mini-instruct_one-year-of-tsla-on-reddit_train-10000-r-778_aps-F-mx-512_lora-16-32-o_proj_qkv_proj-0.01-True_5e-05-linear-0.01-5"
                        f"_seed-1234_ckpt-{ck}_task=causal_lm_dr=defaults",
                        label=f"Phi-3.5-mini-instruct fine-tuned on Reddit gs={ck}",
                    ),
                ],
            )
        return first, seconds

    if base_model_mode is BaseModelMode.PHI_35_MINI_INSTRUCT_FOR_LUSTER_MODELS:
        first = ModelSpec(path="model=Phi-3.5-mini-instruct_task=causal_lm_dr=defaults", label="Phi-3.5-mini-instruct")
        seconds = [
            ModelSpec(path="model=luster-base_task=causal_lm_dr=defaults", label="LUSTER base model"),
            ModelSpec(path="model=luster-base-emotion_task=causal_lm_dr=defaults", label="LUSTER base emotion model"),
            ModelSpec(path="model=luster-chitchat_task=causal_lm_dr=defaults", label="LUSTER chitchat model"),
            ModelSpec(path="model=luster-full_task=causal_lm_dr=defaults", label="LUSTER full model"),
            ModelSpec(path="model=luster-rl-sent_task=causal_lm_dr=defaults", label="LUSTER RL sentiment model"),
            ModelSpec(path="model=luster-rl-succ_task=causal_lm_dr=defaults", label="LUSTER RL success model"),
        ]
        return first, seconds

    if base_model_mode in (BaseModelMode.LLAMA_32_1B, BaseModelMode.LLAMA_32_3B, BaseModelMode.LLAMA_31_8B):
        base = {
            BaseModelMode.LLAMA_32_1B: "Llama-3.2-1B",
            BaseModelMode.LLAMA_32_3B: "Llama-3.2-3B",
            BaseModelMode.LLAMA_31_8B: "Llama-3.1-8B",
        }[base_model_mode]
        first = ModelSpec(f"model={base}_task=causal_lm_dr=defaults", base)
        seconds: list[ModelSpec] = []
        for ck in [800]:
            seconds.extend(
                [
                    ModelSpec(
                        path=f"model={base}-causal_lm-defaults_multiwoz21-r-T-dn-ner_tags_tr-10000-r-778_aps-F-mx-512_lora-16-32-o_proj_q_proj_k_proj_v_proj-0.01-T_5e-05-linear-0.01-f-None-5"
                        f"_seed-1234_ckpt-{ck}_task=causal_lm_dr=defaults",
                        label=f"{base} fine-tuned on MultiWOZ gs={ck}",
                    ),
                    ModelSpec(
                        path=f"model={base}-causal_lm-defaults_one-year-of-tsla-on-reddit-r-T-pr-T-0-0.8-0.1-0.1-ner_tags_tr-10000-r-778_aps-F-mx-512_lora-16-32-o_proj_q_proj_k_proj_v_proj-0.01-T_5e-05-linear-0.01-f-None-5"
                        f"_seed-1234_ckpt-{ck}_task=causal_lm_dr=defaults",
                        label=f"{base} fine-tuned on Reddit gs={ck}",
                    ),
                ],
            )
        return first, seconds

    msg: str = f"Unsupported base model mode: {base_model_mode=}"
    raise ValueError(msg)


def model_modes_for_group(group: str) -> list[BaseModelMode]:
    """Map CLI group choice to a set of base model modes.

    Args:
        group: CLI string for grouping.

    Returns:
        List of BaseModelMode values.

    Raises:
        ValueError: If group is unsupported.

    """
    if group == "all":
        return list(BaseModelMode)
    if group == "phi":
        return [BaseModelMode.PHI_35_MINI_INSTRUCT]
    if group == "phi_luster":
        return [BaseModelMode.PHI_35_MINI_INSTRUCT_FOR_LUSTER_MODELS]
    if group == "llama":
        return [BaseModelMode.LLAMA_32_1B, BaseModelMode.LLAMA_32_3B, BaseModelMode.LLAMA_31_8B]
    msg = f"Unsupported model group: {group=}"
    raise ValueError(msg)


# --------------------------------- Pipeline ------------------------------------


class ViolinPipeline:
    """Coordinates task enumeration, loading, plotting, and reporting."""

    def __init__(self, cfg: RunConfig, logger: logging.Logger | None = None) -> None:
        """Initialize the pipeline.

        Args:
            cfg: RunConfig with all runtime settings.
            logger: Optional logger instance.

        """
        self.cfg = cfg
        self.logger = logger or default_logger
        self.paths = PathBuilder(cfg)
        self.saved_paths: list[pathlib.Path] = []

    # ------------------------ orchestration & tasking -------------------------

    def enumerate_tasks(self) -> Iterable[Task]:
        """Yield all Task instances from the cartesian product + registry."""
        for mode in model_modes_for_group(self.cfg.model_group):
            first, seconds = model_specs_for(mode)
            product_iter = itertools.product(
                self.cfg.dataset_name_choices,
                self.cfg.split_choices,
                self.cfg.samples_choices,
                self.cfg.sampling_mode_choices,
                self.cfg.edh_mode_choices,
                seconds,
            )
            for dataset_name, split, samples, sampling_mode, edh_mode, second in product_iter:
                yield Task(
                    dataset_name=dataset_name,
                    split=split,
                    samples=samples,
                    sampling_mode=sampling_mode,
                    edh_mode=edh_mode,
                    first_model=first,
                    second_model=second,
                )

    def run(self) -> None:
        """Execute the pipeline end-to-end."""
        tasks = list(self.enumerate_tasks())
        with tqdm(total=len(tasks), desc="Generating violin plots") as bar:
            for task in tasks:
                self.process_task(task)
                bar.update(1)
        self._print_grouped_paths()

    # ------------------------------ per-task work -----------------------------

    def process_task(self, task: Task) -> None:
        """Load arrays, produce plot, save results for one task.

        Args:
            task: Task describing dataset/model combination.

        """
        split_and_subsample = self.paths.split_and_subsample_path(
            task.split,
            task.samples,
            task.sampling_mode,
        )
        base_dir_1 = self.paths.model_base_dir(
            task.dataset_name,
            split_and_subsample,
            task.edh_mode,
            task.first_model.path,
        )
        base_dir_2 = self.paths.model_base_dir(
            task.dataset_name,
            split_and_subsample,
            task.edh_mode,
            task.second_model.path,
        )

        file_1 = self.paths.estimates_file(base_dir_1)
        file_2 = self.paths.estimates_file(base_dir_2)

        if self.cfg.verbosity >= Verbosity.NORMAL:
            print(  # noqa: T201 - we want this script to print
                f"file_path_1:\n{file_1}",
            )
            print(  # noqa: T201 - we want this script to print
                f"file_path_2:\n{file_2}",
            )

        try:
            arr1 = np.load(file_1)
            arr2 = np.load(file_2)
        except FileNotFoundError as e:
            print(  # noqa: T201 - we want this script to print
                f"File not found for Dataset: {task.dataset_name=}; "
                f"{task.edh_mode=}; "
                f"Second Model: {task.second_model.path}: {e}",
            )
            return
        except Exception as e:  # noqa: BLE001
            print(  # noqa: T201 - we want this script to print
                f"An error occurred for Dataset: {task.dataset_name}; "
                f"{task.edh_mode=}; "
                f"Second Model: {task.second_model.path}: {e}",
            )
            return

        save_dir = self.paths.plot_save_dir(
            second_model=task.second_model,
            dataset_name=task.dataset_name,
            edh_mode=task.edh_mode,
            split_and_subsample=split_and_subsample,
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        saved = self._plot_and_save(
            arrays=[arr1, arr2],
            labels=[task.first_model.label, task.second_model.label],
            save_dir=save_dir,
        )
        self.saved_paths.extend(saved)

        if self.cfg.save_arrays_in_output_dir:
            out1 = save_dir / "local_estimates_pointwise_array_1.npy"
            out2 = save_dir / "local_estimates_pointwise_array_2.npy"
            if self.cfg.verbosity >= Verbosity.NORMAL:
                print(  # noqa: T201 - we want this script to print
                    f"Saving data arrays to:\n{out1=}\n{out2=}",
                )
            np.save(out1, arr1)
            np.save(out2, arr2)

    # ------------------------------- plotting --------------------------------

    def _plot_and_save(
        self,
        arrays: list[np.ndarray],
        labels: list[str],
        save_dir: pathlib.Path,
    ) -> list[pathlib.Path]:
        """Create violin plot, add legend with stats, and save in all formats.

        Args:
            arrays: Two arrays to compare (first vs second model).
            labels: Labels for x-axis.
            save_dir: Directory for output files.

        Returns:
            List of saved file paths.

        """
        plt.figure(figsize=self.cfg.plot.figsize)

        # Keep parity with original call signature
        sns.violinplot(
            data=arrays,
            density_norm="width",
            inner="quartile",
            split=True,
        )

        plt.ylabel("TwoNN", fontsize=self.cfg.plot.fontsize)
        plt.xticks(ticks=range(len(labels)), labels=labels, fontsize=self.cfg.plot.fontsize)
        plt.yticks(fontsize=self.cfg.plot.fontsize)

        # Pull colors from the drawn violins
        colors = [
            violin.get_facecolor().mean(axis=0)  # type: ignore[attr-defined]
            for violin in plt.gca().collections[: len(arrays)]
        ]
        legend_handles = [
            mpatches.Patch(
                color=colors[i],
                label=(
                    f"Mean={np.mean(arrays[i]):.2f}; "
                    f"Median={np.median(arrays[i]):.2f}; "
                    f"Std={np.std(arrays[i], ddof=self.cfg.ddof):.2f}"
                ),
            )
            for i in range(len(labels))
        ]
        plt.legend(handles=legend_handles, loc="upper right", fontsize=self.cfg.plot.legend_fontsize)

        plt.tight_layout(pad=0.1)

        saved_paths: list[pathlib.Path] = []
        for fmt in self.cfg.plot.formats:
            out = save_dir / f"violin_plot.{fmt.value}"
            if self.cfg.verbosity >= Verbosity.NORMAL:
                print(  # noqa: T201 - we want this script to print
                    f"Saving plot to: {out=}",
                )
            plt.savefig(out, format=fmt.value, bbox_inches="tight")
            saved_paths.append(out)

        plt.close()
        return saved_paths

    # ------------------------------ reporting --------------------------------

    def _print_grouped_paths(self) -> None:
        """Group saved paths by exact 'model=...' segment and print a summary."""
        grouped: dict[str, list[pathlib.Path]] = defaultdict(list)
        for p in self.saved_paths:
            for part in p.parts:
                if part.startswith("model="):  # exact match to loading path segment
                    grouped[part].append(p)
                    break

        print(  # noqa: T201 - we want this script to print
            "\nRelative paths of saved plots (grouped by second_model):",
        )
        for model_part, paths in grouped.items():
            print(  # noqa: T201 - we want this script to print
                f"\nPaths for {model_part}:",
            )
            for path in paths:
                # Preserve your original prefix convention for downstream tools.
                print(  # noqa: T201 - we want this script to print
                    "data/twonn_violin_plots/" + str(path),
                )


# ----------------------------------- CLI --------------------------------------


@click.command()
@click.option(
    "--model-group",
    type=click.Choice(
        choices=[
            "all",
            "phi",
            "phi_luster",
            "llama",
        ],
    ),
    default="all",
    help="Select base model group.",
)
def main(model_group: str) -> None:
    """CLI entrypoint: build config and run the pipeline."""
    cfg = RunConfig(model_group=model_group)
    pipeline = ViolinPipeline(cfg=cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
