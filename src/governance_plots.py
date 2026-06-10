import textwrap
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.cluster.hierarchy import dendrogram, linkage

from dataloader import IndividualProjectContext
from governance_stats import (
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
    base_font_size: int,
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
                f"Trend Analysis: {dim}",
                fontsize=base_font_size,
                fontweight="bold",
                pad=15,
            )
            ax.set_xlabel("Year", fontsize=base_font_size - 2)
            ax.set_ylabel("Score Profile Index", fontsize=base_font_size - 2)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis="both", which="major", labelsize=base_font_size - 3)

            if has_data:
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1),
                    frameon=True,
                    fontsize=base_font_size - 2,
                )

            plt.tight_layout()
            safe_filename = "".join(c if c.isalnum() else "_" for c in dim).lower()
            plt.savefig(output_dir / f"line_{safe_filename}.svg", bbox_inches="tight")
        finally:
            plt.close(fig)


def plot_combined_heatmap(
    projects_stats: List[GovernanceProjectStats],
    output_dir: Path,
    dimensions: List[str],
    base_font_size: int,
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
        ax.set_xticklabels(
            wrapped_labels, rotation=0, ha="center", fontsize=base_font_size - 3
        )
        ax.set_yticklabels(unique_projects, fontsize=base_font_size - 2)

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
                    fontsize=base_font_size - 3,
                )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(
            "Index over 5 years", size=base_font_size - 2
        )
        cbar.ax.tick_params(labelsize=base_font_size - 3)

        ax.set_title(
            "Governance Dimensions Heatmap (5-Year Pooled)",
            fontsize=base_font_size + 1,
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
    base_font_size: int,
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
            fontsize=base_font_size - 3,
            fontweight="bold",
        )
        ax.tick_params(axis="y", labelsize=base_font_size - 3)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            title="Evaluated Projects",
            title_fontsize=base_font_size - 2,
            fontsize=base_font_size - 3,
        )
        ax.set_title(
            "Governance Dimensions Parallel Coordinates (5-Year Pooled)",
            fontsize=base_font_size + 1,
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
    base_font_size: int,
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
            ax.set_xticklabels(
                labels, color="#2c3e50", size=base_font_size - 3, weight="bold"
            )
            ax.set_ylim(0, 1.0)
            ax.tick_params(axis="x", pad=22)

            ax.plot(closed_angles, closed_values, color="#1f77b4", linewidth=2)
            ax.fill(closed_angles, closed_values, color="#1f77b4", alpha=0.18)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(
                ["0.2", "0.4", "0.6", "0.8", "1.0"],
                color="grey",
                size=base_font_size - 3,
            )
            ax.set_title(
                f"{p_obj.project_name} - Governance Profile (5-Year Pooled)",
                fontsize=base_font_size + 1,
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
    base_font_size: int,
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
        ax.set_xticklabels(
            wrapped_labels, rotation=45, ha="right", fontsize=base_font_size - 3
        )
        ax.set_yticklabels(wrapped_labels, fontsize=base_font_size - 3)

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
                    fontsize=base_font_size - 3,
                )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Pearson Correlation Coefficient", size=base_font_size - 2)
        cbar.ax.tick_params(labelsize=base_font_size - 3)

        ax.set_title(
            "Governance Dimensions Correlation Matrix",
            fontsize=base_font_size + 1,
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


def display_validation_results(
    validation_result: KnownGroupsValidationResult,
    output_dir: Path,
    base_font_size: int,
) -> None:
    """
    Consumes a KnownGroupsValidationResult data structure container to print a cleanly formatted
    markdown summary report table and generate publication-quality evaluation boxplots.
    """
    # 1. Parse and print the markdown metrics report table
    df_table = pl.DataFrame(validation_result.validity_rows)
    with pl.Config(
        tbl_formatting="markdown",
        tbl_hide_dataframe_shape=True,
        tbl_width_chars=10000,
        tbl_hide_column_data_types=True,
        tbl_rows=-1,
        tbl_cols=-1,
    ):
        (output_dir / "governance_validation.md").write_text(
            str(df_table), encoding="utf-8"
        )

    # 2. Configure and draw boxplot grid layouts based on identified dimensions
    num_dims = len(validation_result.dimensions)
    if num_dims == 0:
        print("No metrics available to plot.")
        return

    cols = min(3, num_dims)
    rows = (num_dims + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows))
    if num_dims == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, dim in enumerate(validation_result.dimensions):
        ax = axes[idx]
        dim_data = validation_result.group_data[dim]

        # Build flat visualization DataFrame tracking custom category groups
        plot_records = []
        for group_name, array_vals in dim_data.items():
            for val in array_vals:
                plot_records.append({"Group": group_name, "Value": val})
        df_plot = pd.DataFrame(plot_records)

        # Extract row metadata values to build informative headers
        row_meta = next(
            r
            for r in validation_result.validity_rows
            if r["Governance Dimension"] == dim
        )
        title_str = (
            f"{dim}\n{row_meta['Statistical Test']} (p={row_meta['p-value']:.3f})"
        )

        # Draw boxplots and overlaid individual data points
        sns.boxplot(
            x="Group",
            y="Value",
            data=df_plot,
            ax=ax,
            palette="Pastel1",
            width=0.4,
            fliersize=0,
        )
        sns.stripplot(
            x="Group",
            y="Value",
            data=df_plot,
            ax=ax,
            color="black",
            alpha=0.6,
            size=6,
            jitter=0.1,
        )

        ax.set_title(title_str, fontsize=base_font_size - 1, fontweight="bold")
        ax.set_xlabel("", fontsize=base_font_size - 2)
        ax.set_ylabel("Metric Score", fontsize=base_font_size - 2)
        ax.tick_params(axis="y", labelsize=base_font_size - 3)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # Rotate text labels dynamically to prevent overlapping
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=15, ha="right", fontsize=base_font_size - 3
        )

    # Prune any unused axes windows from the grid layout
    for remaining_idx in range(num_dims, len(axes)):
        fig.delaxes(axes[remaining_idx])

    plt.tight_layout()
    plt.savefig(output_dir / "governance_groups_validation.svg", bbox_inches="tight")
    plt.close()


def plot_governance_dendrogram(
    projects_stats: List[GovernanceProjectStats],
    output_dir: Path,
    dimensions: List[str],
    base_font_size: int,
) -> None:
    """
    Applies agglomerative hierarchical clustering on project profiles
    utilizing their true 5-year pooled vectors, rendering a dendrogram map
    and generating an audit markdown detailing the linkage history.
    """
    if not projects_stats or not dimensions:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    unique_projects = sorted([p.project_name for p in projects_stats])

    # 1. Structure the project-by-dimension vector space matrix
    matrix_data = np.zeros((len(unique_projects), len(dimensions)))
    project_map = {p.project_name: p for p in projects_stats}

    for d_idx, dim in enumerate(dimensions):
        for p_idx, p_name in enumerate(unique_projects):
            matrix_data[p_idx, d_idx] = project_map[p_name].pooled_metrics.get(dim, 0.0)

    # 2. Compute Agglomerative Linkage using Ward's minimum variance algorithm
    # rows: samples (projects), columns: features (dimensions)
    Z = linkage(matrix_data, method="ward")

    # 3. Build and save the publication-quality Dendrogram Vector Image
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        dendrogram(
            Z,
            labels=unique_projects,
            orientation="top",
            leaf_rotation=45,
            leaf_font_size=base_font_size - 2,
            ax=ax,
        )
        ax.set_title(
            "Hierarchical Structural Taxonomy (Ward's Linkage)",
            fontsize=base_font_size + 1,
            fontweight="bold",
            pad=15,
        )
        ax.set_ylabel(
            "Cophetic Distance", fontsize=base_font_size - 2
        )
        ax.tick_params(axis="y", labelsize=base_font_size - 3)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(
            output_dir / "governance_hierarchy_dendrogram.svg", bbox_inches="tight"
        )
    finally:
        plt.close(fig)

    # 4. Generate the step-by-step Taxonomy Aggregation Tree Markdown Report
    markdown_path = output_dir / "governance_hierarchy_taxonomy.md"

    md_lines = [
        "# Hierarchical Agglomerative Clustering Audit Trail",
        "\nThis report logs the variance minimization distance progression during bottom-up tree structural building calculations.\n",
        "| Step | Target Cluster A | Target Cluster B | Linkage Distance Threshold | Formed Leaf/Node Cluster Size |",
        "| :---: | :--- | :--- | :---: | :---: |",
    ]

    current_node_names = list(unique_projects)
    num_leaves = len(unique_projects)

    for i, row in enumerate(Z):
        idx_a, idx_b, distance, cluster_size = (
            int(row[0]),
            int(row[1]),
            row[2],
            int(row[3]),
        )

        name_a = current_node_names[idx_a]
        name_b = current_node_names[idx_b]

        new_node_name = f"Node_Cluster_{num_leaves + i} ({name_a} + {name_b})"
        current_node_names.append(new_node_name)

        md_lines.append(
            f"| {i + 1} | {name_a} | {name_b} | {distance:.4f} | {cluster_size} |"
        )

    markdown_path.write_text("\n".join(md_lines), encoding="utf-8")


def show_governance_statistics(
    projects: List[IndividualProjectContext],
    output_dir: Path,
    base_font_size: int = 19,
) -> None:
    """Calculates flat trend lines and robust multi-year pooled profiles over all data assets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    project_records, ordered_keys, known_groups_result = get_governance_statistics(
        projects
    )
    dimensions = _get_all_dimensions(project_records, ordered_keys)

    print("Executing visualization generation tasks over grouped structural records...")

    # Passing the base_font_size down the chain to all visualization functions
    plot_consolidated_line_charts(
        project_records, output_dir, dimensions, base_font_size
    )
    plot_combined_heatmap(project_records, output_dir, dimensions, base_font_size)
    plot_combined_parallel_coordinates(
        project_records, output_dir, dimensions, base_font_size
    )
    plot_project_radars(project_records, output_dir, dimensions, base_font_size)
    plot_dimensions_correlation(project_records, output_dir, dimensions, base_font_size)
    display_validation_results(known_groups_result, output_dir, base_font_size)
    plot_governance_dendrogram(project_records, output_dir, dimensions, base_font_size)

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
