import textwrap
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance import pdist

# Targets the updated dataclass layout
from governance_calc import (
    GovernanceProjectStats,
)
from governance_stats import (
    CorrelationResults,
    _get_metric_data,
    _resolve_attr_name,
    calculate_dimensions_correlation,
)


def plot_consolidated_line_charts(
    projects_stats: List[GovernanceProjectStats],
    output_dir: Path,
    dimensions: List[str],
    base_font_size: int,
) -> None:
    """Plots accurate trends displaying continuous line charts overlaid with 95% CI bands."""
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.colormaps["tab10"](np.linspace(0, 1, max(10, len(projects_stats))))

    for dim in dimensions:
        fig, ax = plt.subplots(figsize=(11, 6))
        try:
            has_data = False
            for idx, p_stat in enumerate(projects_stats):
                attr_name = _resolve_attr_name(dim)
                timeline_profile = getattr(p_stat.metrics, attr_name, None)

                if timeline_profile and timeline_profile.windows:
                    has_data = True
                    years = sorted(timeline_profile.windows.keys())

                    vals = [timeline_profile.windows[y].val for y in years]
                    lows = [timeline_profile.windows[y].ci_low for y in years]
                    highs = [timeline_profile.windows[y].ci_high for y in years]

                    color = colors[idx % len(colors)]

                    # Core Point Estimation Line
                    ax.plot(
                        years,
                        vals,
                        marker="o",
                        linewidth=2,
                        color=color,
                        label=p_stat.project_name,
                        zorder=3,
                    )
                    # Confidence Interval Ribbon Area
                    ax.fill_between(
                        years, lows, highs, color=color, alpha=0.15, zorder=2
                    )

            ax.set_title(
                f"Trend Analysis: {dim} (with 95% CI)",
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
    """Generates cross-project summary maps letting Seaborn completely handle the styling, colors, and text contrast."""
    if not projects_stats or not dimensions:
        return

    normal_text_size = base_font_size - 7

    output_dir.mkdir(parents=True, exist_ok=True)
    unique_projects = sorted([p.project_name for p in projects_stats])
    project_map = {p.project_name: p for p in projects_stats}

    # Build the DataFrames for values and custom text layers
    matrix_data = np.zeros((len(unique_projects), len(dimensions)))

    # Store CI values for our coordinate plotting step later
    ci_bounds = {}

    for d_idx, dim in enumerate(dimensions):
        for p_idx, p_name in enumerate(unique_projects):
            interval = _get_metric_data(project_map[p_name], dim, mode="pooled")
            matrix_data[p_idx, d_idx] = interval.val

            upper_err = interval.ci_high - interval.val
            lower_err = interval.val - interval.ci_low
            ci_bounds[(p_idx, d_idx)] = (lower_err, upper_err)

    wrapped_labels = [textwrap.fill(dim, width=15) for dim in dimensions]
    df_data = pd.DataFrame(matrix_data, index=unique_projects, columns=wrapped_labels)

    fig, ax = plt.subplots(figsize=(11, 7))
    try:
        # Step 1: Let Seaborn map the background colors and render the main text dead-center
        sns.heatmap(
            df_data,
            annot=True,
            fmt=".2f",
            vmin=0.0,
            vmax=1.0,
            ax=ax,
            annot_kws={"fontsize": normal_text_size, "fontweight": "bold"},
            cbar_kws={"label": "Index over 5 years (with 95% CI)"},
        )

        # Step 2: Extract text objects from Seaborn to find the auto-calculated text color
        # This keeps the styling automatic even if we manually draw the CIs below them
        text_objects = [t for t in ax.texts]

        text_idx = 0
        for i in range(len(unique_projects)):
            for j in range(len(dimensions)):
                # Snatch the color Seaborn picked for this specific grid coordinate
                native_color = text_objects[text_idx].get_color()
                text_idx += 1

                lower_err, upper_err = ci_bounds[(i, j)]
                ci_text = f"-{lower_err:.2f} / +{upper_err:.2f}"

                # Step 3: Draw the smaller CI layer directly below the center point
                ax.text(
                    j + 0.5,  # Centered horizontally
                    i + 0.78,  # Dropped down past the core number
                    ci_text,
                    ha="center",
                    va="center",
                    color=native_color,  # Inherits contrast automatically
                    fontweight="normal",
                    fontsize=base_font_size - 12,
                )

        # Labels formatting
        ax.set_xticklabels(
            wrapped_labels, rotation=0, ha="center", fontsize=normal_text_size
        )
        ax.set_yticklabels(unique_projects, rotation=0, fontsize=normal_text_size)

        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_size(normal_text_size)
        cbar.ax.tick_params(labelsize=normal_text_size)

        ax.set_title(
            "Governance Dimensions Heatmap (5-Year Pooled Matrix)",
            fontsize=base_font_size + 1,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout()

        output_file = output_dir / "combined_grouped_heatmap.svg"
        plt.savefig(output_file, bbox_inches="tight")
        print(f"Heatmap successfully rendered and saved to: {output_file}")
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
            vector = [
                _get_metric_data(p_obj, dim, mode="pooled").val for dim in dimensions
            ]
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
        values = [_get_metric_data(p_obj, dim, mode="pooled").val for dim in dimensions]
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
    projects_stats: List["GovernanceProjectStats"],
    output_dir: Path,
    dimensions: List[str],
    base_font_size: int,
) -> None:
    """Calculates correlations and renders visual matrices accompanied by markdown logs."""
    if len(dimensions) < 2 or not projects_stats:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    normal_text_size = base_font_size - 3

    # Step 1: Run the calculation engine (now returning our strongly-typed dict)
    correlation_data: CorrelationResults = calculate_dimensions_correlation(
        projects_stats, dimensions
    )
    if not correlation_data:
        return

    # Labels and layout configurations
    wrapped_labels = [textwrap.fill(dim, width=15) for dim in dimensions]
    mask = np.triu(np.ones((len(dimensions), len(dimensions)), dtype=bool), k=0)

    labels_map = {
        "pearson": "Pearson Correlation Coefficient",
        "spearman": "Spearman Rank Correlation",
    }

    for method, payload in correlation_data.items():
        # Clean attribute access replacing legacy dict string lookups
        corr_df = payload.corr_df
        ci_bounds = payload.ci_bounds
        corr_matrix = corr_df.values
        cbar_label = labels_map[method]

        # Rename columns/index for clean plotting display
        df_plot = corr_df.copy()
        df_plot.columns = wrapped_labels
        df_plot.index = wrapped_labels

        # --- Visual Heatmap Generation ---
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            df_plot,
            mask=mask,
            cmap="coolwarm",
            vmax=1.0,
            vmin=-1.0,
            center=0,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": normal_text_size, "fontweight": "bold"},
            cbar_kws={"label": f"{cbar_label} (with 95% CI)", "shrink": 0.8},
            ax=ax,
        )

        # Snatch Seaborn's text elements to dynamically clone auto-contrasting colors
        text_objects = [t for t in ax.texts]

        text_idx = 0
        for i in range(len(dimensions)):
            for j in range(len(dimensions)):
                # If it's masked or on the diagonal, Seaborn skips drawing text,
                # so we only track active text fields in the lower triangle
                if i > j:
                    native_color = text_objects[text_idx].get_color()
                    text_idx += 1

                    lower_err, upper_err = ci_bounds[(i, j)]
                    ci_text = f"-{lower_err:.2f} / +{upper_err:.2f}"

                    # Write the smaller CI adjustment string under the main value text layer
                    ax.text(
                        j + 0.5,
                        i + 0.76,
                        ci_text,
                        ha="center",
                        va="center",
                        color=native_color,
                        fontweight="normal",
                        fontsize=base_font_size - 10,
                    )

        # Style labels and axes dynamically mapping text sizes
        ax.set_xticklabels(
            wrapped_labels, rotation=45, ha="right", fontsize=normal_text_size
        )
        ax.set_yticklabels(wrapped_labels, rotation=0, fontsize=normal_text_size)

        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_size(base_font_size - 2)
        cbar.ax.tick_params(labelsize=normal_text_size)

        ax.set_title(
            "Governance Dimensions Correlation",
            fontsize=base_font_size + 1,
            fontweight="bold",
            pad=20,
        )

        plt.tight_layout()
        plt.savefig(output_dir / f"dimensions_{method}_matrix.svg", bbox_inches="tight")
        plt.close(fig)

        # --- Markdown Table Generation ---
        markdown_path = output_dir / f"dimensions_{method}_matrix.md"
        md_lines = [
            "# Governance Dimensions Correlation\n",
            f"Type: {method.capitalize()}\n",
            "> Values shown as: **Correlation Coefficient [-Lower CI / +Upper CI]**\n",
            "| Dimension | " + " | ".join(dimensions) + " |",
            "| :--- | " + " | ".join([":---:"] * len(dimensions)) + " |",
        ]

        for i, dim_row in enumerate(dimensions):
            cells = [dim_row]
            for j in range(len(dimensions)):
                if i > j:
                    r = corr_matrix[i, j]
                    l_err, u_err = ci_bounds[(i, j)]
                    cells.append(f"{r:.2f} [-{l_err:.2f}/+{u_err:.2f}]")
                else:
                    cells.append("")
            md_lines.append("| " + " | ".join(cells) + " |")

        markdown_path.write_text("\n".join(md_lines), encoding="utf-8")


def plot_governance_dendrogram(
    projects_stats: List[GovernanceProjectStats],
    output_dir: Path,
    dimensions: List[str],
    base_font_size: int,
) -> None:
    """Applies agglomerative hierarchical clustering on project profiles utilizing true pooled vectors."""
    if not projects_stats or not dimensions:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    unique_projects = sorted([p.project_name for p in projects_stats])

    matrix_data = np.zeros((len(unique_projects), len(dimensions)))
    project_map = {p.project_name: p for p in projects_stats}

    for d_idx, dim in enumerate(dimensions):
        for p_idx, p_name in enumerate(unique_projects):
            matrix_data[p_idx, d_idx] = _get_metric_data(
                project_map[p_name], dim, mode="pooled"
            ).val

    Z = linkage(matrix_data, method="ward")

    # 1. Calculate original pairwise distances (defaults to Euclidean)
    orign_distances = pdist(matrix_data)

    # 2. Calculate the cophenetic correlation coefficient
    coph_corr, coph_dists = cophenet(Z, orign_distances)

    with open(output_dir / "governance_hierarchy_dendrogram_cophenetic.txt", "w") as f:
        f.write(f"Cophenetic Correlation Coefficient: {coph_corr:.4f}\n")

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
        ax.set_ylabel("Cophetic Distance", fontsize=base_font_size - 2)
        ax.tick_params(axis="y", labelsize=base_font_size - 3)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(
            output_dir / "governance_hierarchy_dendrogram.svg", bbox_inches="tight"
        )
    finally:
        plt.close(fig)

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


def plot_projects_2d(
    projects_stats: List[GovernanceProjectStats],
    output_dir: Path,
    dimensions: List[str],
    base_font_size: int,
) -> None:
    """Projects projects into 2D space using PCA (via SVD) and renders a scatter plot."""
    if len(projects_stats) < 2 or not dimensions:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    unique_projects = sorted([p.project_name for p in projects_stats])

    matrix_data = np.zeros((len(unique_projects), len(dimensions)))
    project_map = {p.project_name: p for p in projects_stats}

    for d_idx, dim in enumerate(dimensions):
        for p_idx, p_name in enumerate(unique_projects):
            matrix_data[p_idx, d_idx] = _get_metric_data(
                project_map[p_name], dim, mode="pooled"
            ).val

    mean_vec = np.mean(matrix_data, axis=0)
    centered_matrix = matrix_data - mean_vec

    coords_2d = np.zeros((len(unique_projects), 2))
    explained_variance_ratio = [0.0, 0.0]

    if not np.allclose(centered_matrix, 0) and len(unique_projects) > 1:
        U, S, Vt = np.linalg.svd(centered_matrix, full_matrices=False)

        num_components = min(2, len(S))
        if num_components > 0:
            coords_2d[:, :num_components] = U[:, :num_components] * S[:num_components]

            total_var = np.sum(S**2)
            if total_var > 0:
                for i in range(num_components):
                    explained_variance_ratio[i] = (S[i] ** 2) / total_var

    fig, ax = plt.subplots(figsize=(11, 8))
    try:
        colors = plt.colormaps["tab10"](
            np.linspace(0, 1, max(10, len(unique_projects)))
        )

        for idx, p_name in enumerate(unique_projects):
            x, y = coords_2d[idx, 0], coords_2d[idx, 1]
            ax.scatter(
                x,
                y,
                s=180,
                color=colors[idx % len(colors)],
                label=p_name,
                alpha=0.85,
                edgecolors="black",
                linewidths=1.5,
                zorder=3,
            )
            ax.text(
                x,
                y,
                f"  {p_name}",
                fontsize=base_font_size - 4,
                va="center",
                ha="left",
                fontweight="semibold",
                zorder=4,
            )

        ax.set_title(
            "Project Governance Space (2D PCA Projection)",
            fontsize=base_font_size + 1,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel(
            f"Principal Component 1 ({explained_variance_ratio[0] * 100:.1f}% Variance)",
            fontsize=base_font_size - 2,
        )
        ax.set_ylabel(
            f"Principal Component 2 ({explained_variance_ratio[1] * 100:.1f}% Variance)",
            fontsize=base_font_size - 2,
        )
        ax.grid(True, linestyle="--", alpha=0.5, zorder=1)
        ax.tick_params(axis="both", which="major", labelsize=base_font_size - 3)

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            title="Projects",
            title_fontsize=base_font_size - 2,
            fontsize=base_font_size - 3,
            frameon=True,
        )

        plt.tight_layout()
        plt.savefig(output_dir / "projects_2d_projection.svg", bbox_inches="tight")
    finally:
        plt.close(fig)


def show_independence_with_and_without_discard(
    project_governance_stats: List[GovernanceProjectStats],
    output_dir: Path,
    base_font_size: int,
) -> None:
    path = output_dir / "independence_comparison.md"
    lines = [
        "# Independence Metric Comparison: With vs. Without Unaffiliated Discarding\n",
        "This report compares the Independence metric calculated with the standard approach (discarding unaffiliated authors) against a modified approach that retains all authors regardless of affiliation status.\n",
        "| Project Name | Independence (Discarding Unaffiliated) | Independence (No Discarding) | Difference |\n",
        "| :--- | :---: | :---: | :---: |\n",
    ]
    for p_stat in project_governance_stats:
        ind_disc = p_stat.pooled_metrics.independence.val
        ind_no_disc = p_stat.pooled_metrics.independence_no_discard.val
        diff = ind_no_disc - ind_disc
        lines.append(
            f"| {p_stat.project_name} | {ind_disc:.4f} | {ind_no_disc:.4f} | {diff:.4f} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def show_governance_in_plots(
    project_governance_stats: List[GovernanceProjectStats],
    output_dir: Path,
    base_font_size: int,
) -> None:
    """Calculates flat trend lines and robust multi-year pooled profiles over all data assets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Static list mapping definition order to maintain deterministic processing loops
    dimensions = [
        "Independence",
        "Pluralism",
        "Representation",
        "Decentralized Decision-Making",
        "Autonomous Participation",
    ]

    print("Executing visualization generation tasks over grouped structural records...")

    plot_consolidated_line_charts(
        project_governance_stats, output_dir, dimensions, base_font_size
    )
    plot_combined_heatmap(
        project_governance_stats, output_dir, dimensions, base_font_size
    )
    plot_combined_parallel_coordinates(
        project_governance_stats, output_dir, dimensions, base_font_size
    )
    plot_project_radars(
        project_governance_stats, output_dir, dimensions, base_font_size
    )
    plot_dimensions_correlation(
        project_governance_stats, output_dir, dimensions, base_font_size
    )
    plot_governance_dendrogram(
        project_governance_stats, output_dir, dimensions, base_font_size
    )
    plot_projects_2d(project_governance_stats, output_dir, dimensions, base_font_size)
    show_independence_with_and_without_discard(
        project_governance_stats, output_dir, base_font_size
    )

    # Output Tables Summary Generation formatting point values cleanly
    table_data = {"Governance Domain Framework Structure": dimensions}
    for p_record in project_governance_stats:
        table_data[p_record.project_name] = [
            f"{_get_metric_data(p_record, dim, mode='pooled').val:.4f}"
            for dim in dimensions
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
