import textwrap
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import MaxNLocator

from dataloader import IndividualProjectContext
from governance import (
    GovernanceProjectStats,
    KnownGroupsValidationResult,
    get_governance_statistics,
)


def _get_all_dimensions(
    projects_stats: List[GovernanceProjectStats], ordered_keys: List[str]
) -> List[str]:
    """Extracts unique governance dimension names in the exact registry definition order."""
    present_dims = {dim for p in projects_stats for dim in p.metrics}
    return [dim for dim in ordered_keys if dim in present_dims]


def plot_consolidated_line_charts(
    projects_stats: List[GovernanceProjectStats],
    output_dir: Path,
    dimensions: List[str],
) -> None:
    """Plots accurate discrete trends utilizing single-year historical context dictionaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.colormaps["tab10"](np.linspace(0, 1, max(10, len(projects_stats))))

    for dim in dimensions:
        fig, ax = plt.subplots(figsize=(11, 6))
        try:
            has_data = False
            for idx, p_stat in enumerate(projects_stats):
                if dim in p_stat.metrics:
                    timeline = p_stat.metrics[dim]
                    if not timeline:
                        continue

                    has_data = True
                    years = sorted(timeline.keys())
                    ax.plot(
                        years,
                        [timeline[y] for y in years],
                        marker="o",
                        linewidth=2,
                        color=colors[idx % len(colors)],
                        label=p_stat.project_name,
                    )

            ax.set_title(
                f"Trend Analysis: {dim}", fontsize=12, fontweight="bold", pad=15
            )
            ax.set_xlabel("Year", fontsize=10)
            ax.set_ylabel("Score Profile Index", fontsize=10)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            if has_data:
                ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)

            plt.tight_layout()
            safe_filename = "".join(c if c.isalnum() else "_" for c in dim).lower()
            plt.savefig(output_dir / f"line_{safe_filename}.svg", bbox_inches="tight")
        finally:
            plt.close(fig)


def plot_combined_heatmap(
    projects_stats: List[GovernanceProjectStats],
    output_dir: Path,
    dimensions: List[str],
) -> None:
    """Generates cross-project summary matrix maps utilizing the precise 5-year pooled indices."""
    if not projects_stats or not dimensions:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    unique_projects = sorted([p.project_name for p in projects_stats])

    matrix_data = np.zeros((len(unique_projects), len(dimensions)))
    project_map = {p.project_name: p for p in projects_stats}

    for d_idx, dim in enumerate(dimensions):
        for p_idx, p_name in enumerate(unique_projects):
            matrix_data[p_idx, d_idx] = project_map[p_name].pooled_metrics.get(dim, 0.0)

    fig, ax = plt.subplots(figsize=(12, 7))
    try:
        im = ax.imshow(matrix_data, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
        wrapped_labels = [textwrap.fill(dim, width=15) for dim in dimensions]

        ax.set_xticks(np.arange(len(dimensions)))
        ax.set_yticks(np.arange(len(unique_projects)))
        ax.set_xticklabels(wrapped_labels, rotation=0, ha="center", fontsize=9)
        ax.set_yticklabels(unique_projects, fontsize=10)

        for i in range(len(unique_projects)):
            for j in range(len(dimensions)):
                val = matrix_data[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="black" if val < 0.7 else "white",
                    fontweight="bold",
                )

        fig.colorbar(im, ax=ax, label="5-Year Pooled Structural Index Performance")
        ax.set_title(
            "Grouped Governance Profile Cross-Heatmap Summary (5-Year Pooled)",
            fontsize=13,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()
        plt.savefig(output_dir / "combined_grouped_heatmap.svg", bbox_inches="tight")
    finally:
        plt.close(fig)


def plot_combined_parallel_coordinates(
    projects_stats: List[GovernanceProjectStats],
    output_dir: Path,
    dimensions: List[str],
) -> None:
    """Constructs a Parallel Coordinates Plot mapping projects using their true pooled 5-year vectors."""
    if not projects_stats or len(dimensions) < 2:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    try:
        x_positions = np.arange(len(dimensions))
        colors = plt.colormaps["Set1"](np.linspace(0, 1, len(projects_stats)))

        for idx, p_obj in enumerate(projects_stats):
            vector = [p_obj.pooled_metrics.get(dim, 0.0) for dim in dimensions]
            ax.plot(
                x_positions,
                vector,
                marker="s",
                linewidth=2.5,
                color=colors[idx],
                label=p_obj.project_name,
                alpha=0.85,
            )

        for pos in x_positions:
            ax.axvline(pos, color="black", linestyle="-", alpha=0.25)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [textwrap.fill(d, width=12) for d in dimensions],
            fontsize=9,
            fontweight="bold",
        )
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(
            loc="upper left", bbox_to_anchor=(1.02, 1), title="Evaluated Projects"
        )
        ax.set_title(
            "Grouped Governance Parallel Coordinates Structural Vector (5-Year Pooled)",
            fontsize=13,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / "combined_grouped_parallel_coordinates.svg",
            bbox_inches="tight",
        )
    finally:
        plt.close(fig)


def plot_project_radars(
    projects_stats: List[GovernanceProjectStats],
    output_dir: Path,
    dimensions: List[str],
) -> None:
    """Generates localized spider radar profiles per project parsing accurate block structures."""
    if not dimensions:
        return

    radar_dir = output_dir / "radar_charts"
    radar_dir.mkdir(parents=True, exist_ok=True)

    num_metrics = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    closed_angles = angles + [angles[0]]
    labels = [textwrap.fill(d, width=12) for d in dimensions]

    for p_obj in projects_stats:
        values = [p_obj.pooled_metrics.get(dim, 0.0) for dim in dimensions]
        closed_values = values + [values[0]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        try:
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles)
            ax.set_xticklabels(labels, color="#2c3e50", size=9, weight="bold")
            ax.set_ylim(0, 1.0)
            ax.tick_params(axis="x", pad=22)

            ax.plot(closed_angles, closed_values, color="#1f77b4", linewidth=2)
            ax.fill(closed_angles, closed_values, color="#1f77b4", alpha=0.18)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(
                ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=9
            )
            ax.set_title(
                f"{p_obj.project_name} - Governance Profile (5-Year Pooled)",
                fontsize=13,
                fontweight="bold",
                pad=30,
                y=1.05,
            )

            safe_name = "".join(
                c if c.isalnum() else "_" for c in p_obj.project_name
            ).lower()
            plt.savefig(
                radar_dir / f"proportional_radar_{safe_name}.svg", bbox_inches="tight"
            )
        finally:
            plt.close(fig)


def plot_dimensions_correlation(
    projects_stats: List[GovernanceProjectStats],
    output_dir: Path,
    dimensions: List[str],
) -> None:
    """Extracts pooled metrics matrix, computes safe correlations, and saves outputs."""
    if len(dimensions) < 2 or not projects_stats:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    unique_projects = sorted([p.project_name for p in projects_stats])

    matrix_data = np.zeros((len(unique_projects), len(dimensions)))
    project_map = {p.project_name: p for p in projects_stats}

    for d_idx, dim in enumerate(dimensions):
        for p_idx, p_name in enumerate(unique_projects):
            matrix_data[p_idx, d_idx] = project_map[p_name].pooled_metrics.get(dim, 0.0)

    # Clean check for Constant features to prevent Pearson division by zero NaN propagation
    std_deviations = np.std(matrix_data, axis=0)
    constant_mask = std_deviations == 0

    min_vals = matrix_data.min(axis=0)
    max_vals = matrix_data.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[constant_mask] = 1.0
    normalized_matrix = (matrix_data - min_vals) / range_vals

    corr_matrix = np.corrcoef(normalized_matrix, rowvar=False)

    # Secure matrix against NaN flags remaining if any dimensions are constant
    if np.isnan(corr_matrix).any():
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        for idx, is_constant in enumerate(constant_mask):
            if is_constant:
                corr_matrix[idx, :] = 0.0
                corr_matrix[:, idx] = 0.0
                corr_matrix[idx, idx] = 1.0

    fig, ax = plt.subplots(figsize=(10, 8))
    try:
        im = ax.imshow(corr_matrix, cmap="RdBu", aspect="auto", vmin=-1.0, vmax=1.0)
        wrapped_labels = [textwrap.fill(dim, width=15) for dim in dimensions]

        ax.set_xticks(np.arange(len(dimensions)))
        ax.set_yticks(np.arange(len(dimensions)))
        ax.set_xticklabels(wrapped_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(wrapped_labels, fontsize=9)

        for i in range(len(dimensions)):
            for j in range(len(dimensions)):
                val = corr_matrix[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="black" if abs(val) < 0.5 else "white",
                    fontweight="bold",
                )

        fig.colorbar(im, ax=ax, label="Pearson Correlation Coefficient")
        ax.set_title(
            "Governance Framework Dimensions Correlation Matrix",
            fontsize=13,
            fontweight="bold",
            pad=20,
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / "dimensions_correlation_matrix.svg", bbox_inches="tight"
        )
    finally:
        plt.close(fig)

    # Document to Markdown
    markdown_path = output_dir / "dimensions_correlation_matrix.md"
    md_lines = ["# Governance Framework Dimensions Correlation Matrix\n"]
    md_lines.append("| Dimension | " + " | ".join(dimensions) + " |")
    md_lines.append("| :--- | " + " | ".join([":---:"] * len(dimensions)) + " |")

    for i, dim_row in enumerate(dimensions):
        row_cells = [dim_row] + [
            f"{corr_matrix[i, j]:.2f}" for j in range(len(dimensions))
        ]
        md_lines.append("| " + " | ".join(row_cells) + " |")

    markdown_path.write_text("\n".join(md_lines), encoding="utf-8")


def plot_known_groups_validity(
    analysis_result: KnownGroupsValidationResult,
    output_dir: Path,
) -> None:
    """Consumes calculation containers to render grid structural validations cleanly."""
    dimensions = analysis_result.dimensions
    if not dimensions:
        print("No valid dimensions found for rendering visualizations.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    try:
        for idx, dim in enumerate(dimensions):
            ax = axes[idx]
            comm_vals = analysis_result.group_data[dim]["Community"]
            corp_vals = analysis_result.group_data[dim]["Corporate"]
            box_data = [comm_vals, corp_vals]

            bp = ax.boxplot(
                box_data,
                patch_artist=True,
                labels=[
                    f"Community\n(N={len(comm_vals)})",
                    f"Corporate\n(N={len(corp_vals)})",
                ],
                widths=0.4,
            )

            colors = ["#4db6ac", "#ff8a80"]
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            for median in bp["medians"]:
                median.set(color="#37474f", linewidth=2)

            for g_idx, vals in enumerate(box_data, start=1):
                x_jitter = np.random.normal(g_idx, 0.04, size=len(vals))
                ax.scatter(
                    x_jitter,
                    vals,
                    color="#212121",
                    alpha=0.8,
                    edgecolor="white",
                    zorder=3,
                    s=40,
                )

            ax.set_title(
                textwrap.fill(dim, width=25), fontsize=11, fontweight="bold", pad=10
            )
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        if len(dimensions) < len(axes):
            for empty_idx in range(len(dimensions), len(axes)):
                fig.delaxes(axes[empty_idx])

        fig.suptitle(
            "Known-Groups Framework Metric Box Plots (Community vs. Corporate)",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / "known_groups_discriminant_validity.svg", bbox_inches="tight"
        )
    finally:
        plt.close(fig)

    validity_df = pl.DataFrame(analysis_result.validity_rows)
    with pl.Config(
        tbl_formatting="markdown",
        tbl_hide_dataframe_shape=True,
        tbl_width_chars=10000,
        tbl_hide_column_data_types=True,
        tbl_rows=-1,
        tbl_cols=-1,
    ):
        (output_dir / "known_groups_validity_report.md").write_text(
            str(validity_df), encoding="utf-8"
        )


def show_governance_statistics(
    projects: List[IndividualProjectContext], output_dir: Path
) -> None:
    """Calculates flat trend lines and robust multi-year pooled profiles over all data assets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    project_records, ordered_keys, known_groups_result = get_governance_statistics(
        projects
    )
    dimensions = _get_all_dimensions(project_records, ordered_keys)

    print("Executing visualization generation tasks over grouped structural records...")
    plot_consolidated_line_charts(project_records, output_dir, dimensions)
    plot_combined_heatmap(project_records, output_dir, dimensions)
    plot_combined_parallel_coordinates(project_records, output_dir, dimensions)
    plot_project_radars(project_records, output_dir, dimensions)
    plot_dimensions_correlation(project_records, output_dir, dimensions)
    plot_known_groups_validity(known_groups_result, output_dir)

    # Output Tables Summary Generation
    table_data = {"Governance Domain Framework Structure": dimensions}
    for p_record in project_records:
        table_data[p_record.project_name] = [
            round(p_record.pooled_metrics.get(dim, 0.0), 4) for dim in dimensions
        ]

    summary_df = pl.DataFrame(table_data)
    with pl.Config(
        tbl_formatting="markdown",
        tbl_hide_dataframe_shape=True,
        tbl_width_chars=10000,
        tbl_hide_column_data_types=True,
        tbl_rows=-1,
        tbl_cols=-1,
    ):
        (output_dir / "governance_statistics.md").write_text(
            str(summary_df), encoding="utf-8"
        )

    print(
        f"Data calculations pipeline terminated successfully. Files written to: {output_dir}"
    )
