from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .models import AnnotationProfile, ChartRecipe, ChartType


def render_chart(recipe: ChartRecipe) -> bytes:
    fig, ax = plt.subplots(
        figsize=recipe.render_plan.figure_size_inches,
        dpi=recipe.render_plan.dpi,
    )
    try:
        if recipe.answer.chart_type == ChartType.LINE:
            _render_line_chart(ax, recipe)
        elif recipe.answer.chart_type == ChartType.BAR:
            _render_bar_chart(ax, recipe)
        else:
            raise ValueError(f"Unsupported chart type: {recipe.answer.chart_type}")

        if recipe.render_plan.title:
            ax.set_title(recipe.render_plan.title)
        if recipe.render_plan.x_label:
            ax.set_xlabel(recipe.render_plan.x_label)
        ax.set_ylabel(recipe.render_plan.y_label)

        if recipe.render_plan.show_grid:
            ax.grid(True, alpha=0.25)
        if recipe.render_plan.rotate_x_labels:
            ax.tick_params(axis="x", rotation=20)

        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def _render_line_chart(ax: plt.Axes, recipe: ChartRecipe) -> None:
    series = recipe.answer.series[0]
    x_values = [float(point.x) for point in series.points]
    y_values = [point.y for point in series.points]
    marker = "o" if recipe.render_plan.show_markers else None
    ax.plot(
        x_values,
        y_values,
        color=recipe.render_plan.color_hex,
        linewidth=recipe.render_plan.line_width,
        marker=marker,
        markersize=recipe.render_plan.marker_size,
    )
    if recipe.spec.annotation_profile == AnnotationProfile.ENDPOINT_LABELS:
        ax.annotate(
            recipe.series_name,
            xy=(x_values[-1], y_values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )


def _render_bar_chart(ax: plt.Axes, recipe: ChartRecipe) -> None:
    series = recipe.answer.series[0]
    x_values = [str(point.x) for point in series.points]
    y_values = [point.y for point in series.points]
    bars = ax.bar(x_values, y_values, color=recipe.render_plan.color_hex)
    if recipe.spec.annotation_profile == AnnotationProfile.ENDPOINT_LABELS:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2.0, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
