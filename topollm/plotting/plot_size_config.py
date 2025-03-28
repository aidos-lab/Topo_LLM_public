"""Config objects for plot sizes and dimensions."""

from dataclasses import dataclass

import matplotlib.axes


@dataclass
class AxisLimits:
    """Container for axis limits."""

    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None

    @property
    def y_range_description(
        self,
    ) -> str:
        """Return a string describing the y-axis range."""
        description: str = f"y_min={self.y_min}_y_max={self.y_max}"

        return description

    def set_y_axis_limits(
        self,
        axis: matplotlib.axes.Axes,
    ) -> matplotlib.axes.Axes:
        """Set y-axis limits on the given axis."""
        if self.y_min is not None:
            axis.set_ylim(
                bottom=self.y_min,
            )
        if self.y_max is not None:
            axis.set_ylim(
                top=self.y_max,
            )

        return axis


@dataclass
class OutputDimensions:
    """Container for output dimensions."""

    output_pdf_width: int = 2500
    output_pdf_height: int = 1500


@dataclass
class PlotSizeConfigNested:
    """Configuration for axis limits and output figure dimensions."""

    primary_axis_limits: AxisLimits
    secondary_axis_limits: AxisLimits
    output_dimensions: OutputDimensions

    @property
    def y_range_description(
        self,
    ) -> str:
        """Return a string describing the y-axis range."""
        description: str = (
            f"primary_{self.primary_axis_limits.y_range_description}"
            f"_secondary_{self.secondary_axis_limits.y_range_description}"
        )

        return description


@dataclass
class PlotSizeConfigFlat:
    """Configuration for axis limits and output figure dimensions.

    Attributes:
        x_min: Minimum value for the x-axis (or None to auto-scale).
        x_max: Maximum value for the x-axis (or None to auto-scale).
        y_min: Minimum value for the y-axis (or None to auto-scale).
        y_max: Maximum value for the y-axis (or None to auto-scale).
        output_pdf_width: Figure width in pixels when saving to PDF.
        output_pdf_height: Figure height in pixels when saving to PDF.

    """

    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    output_pdf_width: int = 2500
    output_pdf_height: int = 1500


@dataclass
class PlotColumnsConfig:
    """Configuration for column names used in the plot.

    Attributes:
        x_column: Name of the column for x-axis values.
        y_column: Name of the column for y-axis values.
        group_column: Name of the categorical column to group the data.
        std_column: Optional name of the column for standard deviation values.
                   If provided and found in the DataFrame, an error band will be plotted.

    """

    x_column: str = "model_checkpoint"
    y_column: str = "loss_mean"
    group_column: str = "data_full"
    std_column: str | None = None
