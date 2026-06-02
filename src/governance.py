"""Governance metrics computation, aggregation, and consolidated single-plot visualization."""

import datetime
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.statistics import IndividualProjectContext


@dataclass
class GovMetricConfig:
    """Configuration mapping for processing, tracking, and plotting governance dynamics."""

    filename: str
    title: str
    compute_fn: Callable[[IndividualProjectContext], Dict[int, float]]


# =====================================================================
# Atomic Mathematical Helpers
# =====================================================================


def _polars_gini(values: pl.Series) -> float:
    """Computes an unbiased Gini Coefficient for an array of values efficiently."""
    x = values.drop_nulls().to_numpy().astype(float)
    if len(x) <= 1 or x.sum() == 0:
        return 0.0

    n = len(x)
    x = np.sort(x)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x) / (n * np.sum(x))) - (n + 1) / n) * (
        n / (n - 1)
    )


# =====================================================================
# Metric Transformation Engines (Slicing Context Safely)
# =====================================================================


def compute_independence_hhi_per_year(
    ctx: IndividualProjectContext,
) -> Dict[int, float]:
    """Computes Normalized Inverse HHI for organizational independence per year.

    Formula: I = 1 - HHI* = 1 - ((HHI - 1/n) / (1 - 1/n))
    """
    first_proposal_date = ctx.proposal_revisions.group_by("proposal_id").agg(
        pl.col("created_at").min()
    )

    df = (
        first_proposal_date.join(
            ctx.proposal_revision_authors, on="proposal_id", how="inner"
        )
        .with_columns(pl.col("created_at").dt.year().alias("year"))
        .filter(pl.col("year").is_not_null())
        .select(["year", "proposal_id", "author_id"])
    )
    if df.is_empty():
        logging.warning(
            f"No valid proposal revisions with authors and timestamps found in {ctx.project_name}."
        )
        return {}

    org_map = ctx.affiliations.join(
        ctx.organisations, on="organisation_id", how="inner"
    )
    fallback_map = ctx.person_identifiers.select(["person_id", "domain"])

    df_with_orgs = (
        df.join(org_map, left_on="author_id", right_on="person_id", how="left")
        .join(fallback_map, left_on="author_id", right_on="person_id", how="left")
        .with_columns(
            pl.coalesce(
                [
                    pl.col("organisation_name"),
                    pl.col("domain"),
                    pl.col("author_id").cast(pl.String),
                ]
            ).alias("org")
        )
    )

    # Group by year to compute HHI component factors
    yearly_data = (
        df_with_orgs.group_by(["year", "org"])
        .agg(pl.col("author_id").n_unique().alias("unique_authors_per_org"))
        .with_columns(
            (
                pl.col("unique_authors_per_org")
                / pl.col("unique_authors_per_org").sum().over("year")
            ).alias("share")
        )
        .group_by("year")
        .agg(
            [
                (pl.col("share") ** 2).sum().alias("raw_hhi"),
                pl.col("org").n_unique().alias("n_orgs"),
            ]
        )
    )

    # Apply LaTeX normalization context: I = 1 - HHI*
    year_independence = {}
    for row in yearly_data.iter_rows(named=True):
        yr = int(row["year"])
        raw_hhi = float(row["raw_hhi"])
        n = int(row["n_orgs"])

        if n <= 1:
            score = 0.0
        else:
            hhi_star = (raw_hhi - (1.0 / n)) / (1.0 - (1.0 / n))
            score = float(1.0 - hhi_star)

        year_independence[yr] = max(0.0, min(1.0, score))

    return year_independence


def compute_pluralism_author_gini_per_year(
    ctx: IndividualProjectContext,
) -> Dict[int, float]:
    """Computes Pluralism Score as the inverse Gini Coefficient of proposal authorships (1 - G_x)."""
    first_revision = ctx.proposal_revisions.group_by("proposal_id").agg(
        [
            pl.col("revision_index").min().alias("min_rev"),
            pl.col("created_at").min().alias("created_at"),
        ]
    )

    df = (
        first_revision.join(
            ctx.proposal_revision_authors,
            left_on=["proposal_id", "min_rev"],
            right_on=["proposal_id", "revision_index"],
            how="inner",
        )
        .with_columns(pl.col("created_at").dt.year().alias("year"))
        .filter(pl.col("year").is_not_null())
    )
    if df.is_empty():
        logging.warning(
            f"No valid proposal revisions with authors and timestamps found in {ctx.project_name}."
        )
        return {}

    author_counts = df.group_by(["year", "author_id"]).agg(
        pl.len().alias("contribution_count")
    )

    # Transform raw Gini index into inverse governance scale (1 - Gini)
    return {
        int(year[0]): float(1.0 - _polars_gini(group["contribution_count"]))
        for year, group in author_counts.partition_by("year", as_dict=True).items()
    }


def compute_representation_comment_gini_per_year(
    ctx: IndividualProjectContext,
) -> Dict[int, float]:
    """Computes Representation Score as the inverse Gini Coefficient of discussion comment loops (1 - G_c)."""
    df = ctx.comments.filter(
        pl.col("author_id").is_not_null() & pl.col("created_at").is_not_null()
    ).with_columns(pl.col("created_at").dt.year().alias("year"))

    if df.is_empty():
        logging.warning(
            f"No valid comments with authors and timestamps found in {ctx.project_name}."
        )
        return {}

    comment_counts = df.group_by(["year", "author_id"]).agg(
        pl.len().alias("comment_count")
    )

    # Transform raw Gini index into inverse governance scale (1 - Gini)
    return {
        int(year[0]): float(1.0 - _polars_gini(group["comment_count"]))
        for year, group in comment_counts.partition_by("year", as_dict=True).items()
    }


def compute_betweenness_centralization_per_year(
    ctx: IndividualProjectContext,
) -> Dict[int, float]:
    """Computes Decentralization Score (1 - C_B) from a co-participation interaction graph."""
    # Track proposals authored or co-authored (Revision steps)
    rev_df = (
        ctx.proposal_revision_authors.join(
            ctx.proposal_revisions, on=["proposal_id", "revision_index"], how="inner"
        )
        .with_columns(
            [
                pl.col("created_at").dt.year().alias("year"),
                pl.col("proposal_id").alias("group_id"),
            ]
        )
        .select(["year", "group_id", "author_id"])
    )

    # Track shared discussion contexts
    comment_df = (
        ctx.comments.filter(
            pl.col("proposal_id").is_not_null()
            & pl.col("author_id").is_not_null()
            & pl.col("created_at").is_not_null()
        )
        .with_columns(
            [
                pl.col("created_at").dt.year().alias("year"),
                pl.col("proposal_id").alias("group_id"),
            ]
        )
        .select(["year", "group_id", "author_id"])
    )

    combined_df = pl.concat([rev_df, comment_df]).filter(
        pl.col("year").is_not_null() & pl.col("author_id").is_not_null()
    )
    if combined_df.is_empty():
        logging.warning(f"No valid combined data found in {ctx.project_name}.")
        return {}

    # Self-join evaluates interaction weight frequencies w(i, j) = r_ij + c_ij
    edges_df = (
        combined_df.join(
            combined_df, on=["year", "group_id"], how="inner", suffix="_right"
        )
        .filter(pl.col("author_id") < pl.col("author_id_right"))
        .group_by(["year", "author_id", "author_id_right"])
        .agg(pl.len().alias("weight"))
        # Distance metric configured inverse to structural weight for path algorithms
        .with_columns((1.0 / pl.col("weight")).alias("distance"))
    )

    year_centralization = {}
    all_years = combined_df["year"].unique().sort().to_list()

    for yr in all_years:
        yr_nodes = (
            combined_df.filter(pl.col("year") == yr)["author_id"].unique().to_list()
        )
        yr_edges = edges_df.filter(pl.col("year") == yr)

        if yr_edges.is_empty():
            logging.warning(
                f"No valid edges found for year {yr} in {ctx.project_name}."
            )
            year_centralization[int(yr)] = 1.0
            continue
        if len(yr_nodes) < 3:
            year_centralization[int(yr)] = 1.0
            continue

        G = nx.Graph()
        G.add_nodes_from(yr_nodes)
        G.add_weighted_edges_from(
            yr_edges.select(["author_id", "author_id_right", "distance"]).iter_rows(),
            weight="distance",
        )

        # Compute edge distance weighted betweenness centrality
        centrality = nx.betweenness_centrality(G, weight="distance", normalized=True)
        if not centrality:
            year_centralization[int(yr)] = 1.0
            continue

        max_cb = max(centrality.values())
        n = G.number_of_nodes()
        sum_diff = sum((max_cb - v) for v in centrality.values())
        denom = (n - 1) * (n - 2)

        cb_index = float(sum_diff / denom if denom > 0 else 0.0)

        # Apply normalization framework context: D = 1 - C_B
        year_centralization[int(yr)] = float(1.0 - cb_index)

    return year_centralization


def compute_newcomers_onboarding_per_year(
    ctx: IndividualProjectContext,
) -> Dict[int, float]:
    """Computes Autonomous Participation Score as the proportion of newcomer proposals."""
    global_earliest = (
        ctx.proposal_revision_authors.join(
            ctx.proposal_revisions, on=["proposal_id", "revision_index"], how="inner"
        )
        .group_by("author_id")
        .agg(pl.col("created_at").min().alias("first_global_activity"))
    )

    first_revision = ctx.proposal_revisions.group_by("proposal_id").agg(
        [
            pl.col("revision_index").min().alias("min_rev"),
            pl.col("created_at").min().alias("proposal_created_at"),
        ]
    )

    proposals_df = (
        first_revision.join(
            ctx.proposal_revision_authors,
            left_on=["proposal_id", "min_rev"],
            right_on=["proposal_id", "revision_index"],
            how="inner",
        )
        .join(global_earliest, on="author_id", how="left")
        .with_columns(pl.col("proposal_created_at").dt.year().alias("year"))
        .filter(pl.col("year").is_not_null())
    )
    if proposals_df.is_empty():
        logging.warning(
            f"No valid proposal revisions with authors and timestamps found in {ctx.project_name}."
        )
        return {}

    analysis_df = (
        proposals_df.with_columns(
            (pl.col("first_global_activity") < pl.col("proposal_created_at")).alias(
                "author_is_experienced"
            )
        )
        .group_by(["year", "proposal_id"])
        .agg(pl.col("author_is_experienced").any().alias("has_experienced_author"))
        .with_columns((~pl.col("has_experienced_author")).alias("is_newcomer_proposal"))
    )

    onboarding_rate = (
        analysis_df.group_by("year")
        .agg(
            [
                pl.col("proposal_id").n_unique().alias("total_proposals"),
                pl.col("is_newcomer_proposal").sum().alias("newcomer_proposals"),
            ]
        )
        .with_columns(
            (pl.col("newcomer_proposals") / pl.col("total_proposals")).alias("rate")
        )
        .sort("year")
    )
    return dict(
        zip(onboarding_rate["year"].to_list(), onboarding_rate["rate"].to_list())
    )


# Registry mapping linking calculations to config properties
GOV_REGISTRY = [
    GovMetricConfig(
        "independence.png",
        "Independence (Inverse HHI)",
        compute_independence_hhi_per_year,
    ),
    GovMetricConfig(
        "pluralism.png",
        "Pluralism (Inverse Gini Author Variety)",
        compute_pluralism_author_gini_per_year,
    ),
    GovMetricConfig(
        "representation.png",
        "Representation (Inverse Gini Comment Concentration)",
        compute_representation_comment_gini_per_year,
    ),
    GovMetricConfig(
        "decentralized_decision_making.png",
        "Decentralized Decision-Making (Inverse Betweenness Centralization)",
        compute_betweenness_centralization_per_year,
    ),
    GovMetricConfig(
        "autonomous_participation.png",
        "Autonomous Participation (Onboarding Rate)",
        compute_newcomers_onboarding_per_year,
    ),
]

# =====================================================================
# UI Engine Rendering Core Mechanics
# =====================================================================


def _render_consolidated_gov_plot(
    contexts: List[IndividualProjectContext], cfg: GovMetricConfig, output_dir: Path
) -> None:
    """Plots lines for all projects into a single unified figure with a distinct color map."""
    fig, ax = plt.subplots(figsize=(11, 6))

    colors = plt.colormaps["tab10"](np.linspace(0, 1, max(10, len(contexts))))

    has_data = False
    for idx, ctx in enumerate(contexts):
        scores_dict = cfg.compute_fn(ctx)
        if not scores_dict:
            continue

        has_data = True
        years = sorted(scores_dict.keys())
        scores = [scores_dict[y] for y in years]

        ax.plot(
            years,
            scores,
            marker="o",
            linewidth=2,
            markersize=6,
            color=colors[idx % len(colors)],
            label=ctx.project_name,
        )

    if not has_data:
        ax.text(
            0.5,
            0.5,
            f"No Data Available for Any Projects\n{cfg.title}",
            fontsize=12,
            weight="bold",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="darkred",
        )
    else:
        ax.legend(
            loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True
        )

    ax.set_title(cfg.title, fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(str(output_dir / cfg.filename), dpi=300, bbox_inches="tight")
    plt.close()


def _get_radar_data(ctx: IndividualProjectContext) -> tuple[list[str], list[float]]:
    """Helper to calculate and extract labels and values for a project context."""
    current_year = datetime.datetime.now().year
    target_years = set(range(current_year - 4, current_year + 1))

    labels = []
    values = []

    for cfg in GOV_REGISTRY:
        labels.append(cfg.title)
        scores_dict = cfg.compute_fn(ctx)
        filtered_scores = [v for k, v in scores_dict.items() if k in target_years]

        if filtered_scores:
            values.append(float(np.mean(filtered_scores)))
        else:
            values.append(0.0)

    return labels, values


def _format_polar_axis(ax: plt.Axes, labels: list[str], angles: list[float]) -> None:
    """Applies consistent styling, ticks, and label positioning to a given polar axis."""
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="grey", size=8)

    # Clean up alignment based on hemisphere orientation
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        angle_deg = np.degrees(angle)
        if 0 < angle_deg < 180:
            label.set_horizontalalignment("left")
        elif 180 < angle_deg < 360:
            label.set_horizontalalignment("right")
        else:
            label.set_horizontalalignment("center")
        label.set_wrap(True)

    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
    ax.set_ylim(0, 1.05)


def _render_single_project_radar(
    ctx: IndividualProjectContext,
    labels: list[str],
    values: list[float],
    output_dir: Path,
) -> None:
    """Generates and saves a single radar chart for one project."""
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    closed_values = values + values[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    _format_polar_axis(ax, labels, angles)

    ax.plot(angles, closed_values, color="#1f77b4", linewidth=2, linestyle="solid")
    ax.fill(angles, closed_values, color="#1f77b4", alpha=0.25)

    ax.set_title(
        f"Governance Profile: {ctx.project_name}\n(5-Year Average)",
        fontsize=11,
        fontweight="bold",
        pad=25,
    )

    safe_project_name = "".join(
        c if c.isalnum() else "_" for c in ctx.project_name
    ).lower()
    filename = f"radar_{safe_project_name}.png"

    plt.tight_layout()
    plt.savefig(str(output_dir / filename), dpi=300, bbox_inches="tight")
    plt.close()


def _render_combined_grid_radar(
    radar_datasets: list[tuple[str, list[str], list[float]]],
    output_dir: Path,
    max_cols: int = 3,
) -> None:
    """Generates a single image containing a grid layout of separate radar charts side-by-side."""
    if not radar_datasets:
        return

    num_projects = len(radar_datasets)
    cols = min(num_projects, max_cols)
    rows = math.ceil(num_projects / cols)

    # Dynamic sizing based on grid dimensions (e.g., 5 inches per subplot)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 5, rows * 5),
        subplot_kw=dict(polar=True),
        squeeze=False,  # Ensures 'axes' is always a 2D array even if 1 row or 1 col
    )

    # Convert standard palette to iterable colors for the grid variation
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (project_name, labels, values) in enumerate(radar_datasets):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        closed_values = values + values[:1]

        # Setup axis styling for this specific subplot grid cell
        _format_polar_axis(ax, labels, angles)

        color = color_cycle[idx % len(color_cycle)]
        ax.plot(angles, closed_values, color=color, linewidth=2, linestyle="solid")
        ax.fill(angles, closed_values, color=color, alpha=0.2)

        ax.set_title(project_name, fontsize=12, fontweight="bold", pad=20)

    # Hide any unused subplots in the grid if the project count doesn't fill the row perfectly
    for idx in range(num_projects, rows * cols):
        row = idx // cols
        col = idx % cols
        fig.delaxes(axes[row, col])

    plt.suptitle(
        "Consolidated Governance Profile Grid (5-Year Average)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        str(output_dir / "radar_combined_grid.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def show_governance_statistics(
    projects: list[IndividualProjectContext], output_dir: Path
) -> None:
    """Calculates all governance indices, producing consolidated multi-line plots,

    individual radar files, and a final grouped grid-layout dashboard.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for cfg in GOV_REGISTRY:
        _render_consolidated_gov_plot(projects, cfg, output_dir)

    all_radar_data = []

    for ctx in projects:
        # 1. Calculate the radar variables
        labels, values = _get_radar_data(ctx)
        all_radar_data.append((ctx.project_name, labels, values))

        # 2. Render and save the isolated individual graph file
        _render_single_project_radar(ctx, labels, values, output_dir)

    # 3. Render and save the matrix/grid containing everyone next to each other
    _render_combined_grid_radar(all_radar_data, output_dir, max_cols=3)
