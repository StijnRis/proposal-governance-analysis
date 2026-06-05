import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import rustworkx as rx
from matplotlib.ticker import MaxNLocator

from src.statistics import IndividualProjectContext

# =====================================================================
# Structured Data Containers
# =====================================================================


@dataclass
class GovernanceStatItem:
    """Represents a structural governance domain category containing one or more metrics.

    Structure of metrics data:
    {
        "Metric/Sub-metric Name": {
            2024: 0.88,
            2025: 0.91
        }
    }
    """

    stat_name: str
    metrics: Dict[str, Dict[int, float]] = field(default_factory=dict)


@dataclass
class GovernanceProjectStats:
    """Top-level container storing all categorized governance item dimensions for a project."""

    project_name: str
    items: Dict[str, GovernanceStatItem] = field(default_factory=dict)

    def get_or_create_item(self, stat_name: str) -> GovernanceStatItem:
        """Safely fetches or initializes an isolated governance stat node."""
        if stat_name not in self.items:
            self.items[stat_name] = GovernanceStatItem(stat_name=stat_name)
        return self.items[stat_name]


# =====================================================================
# Atomic Mathematical Helpers
# =====================================================================


def _polars_gini_expr(col_name: str) -> pl.Expr:
    """Returns a Polars expression calculating an unbiased Gini Coefficient."""
    valid_sorted = pl.col(col_name).drop_nulls().sort()
    n = valid_sorted.len()
    sum_x = valid_sorted.sum()

    index = valid_sorted.rank("ordinal")
    gini_raw = (2 * (index * valid_sorted).sum() / (n * sum_x)) - ((n + 1) / n)
    unbiased_gini = gini_raw * (n / (n - 1))

    return pl.when((n <= 1) | (sum_x == 0)).then(0.0).otherwise(unbiased_gini)


# =====================================================================
# Metric Transformation Engines
# =====================================================================


def compute_independence_hhi_per_year(
    ctx: IndividualProjectContext,
) -> Dict[int, Dict[str, float]]:
    """Computes Normalized Inverse HHI for organizational independence per year.

    Excludes authors who are not affiliated with any registered organization.
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
        logging.warning(f"No valid proposal revisions found in {ctx.project_name}.")
        return {}

    # Map authors to their exact organizations
    org_map = ctx.affiliations.join(
        ctx.organisations, on="organisation_id", how="inner"
    )

    # Change to how="inner" to auto-drop any author without an official organization affiliation
    df_with_orgs = df.join(
        org_map, left_on="author_id", right_on="person_id", how="inner"
    ).with_columns(pl.col("organisation_name").alias("org"))

    if df_with_orgs.height < 10:
        logging.warning(
            f"Insufficient data for {ctx.project_name}. "
            f"Found only {df_with_orgs.height} corporate affiliations (minimum 10 required)."
        )
        return {}

    # If all authors were filtered out because none have affiliations, return empty
    if df_with_orgs.is_empty():
        logging.warning(f"No affiliated authors found in {ctx.project_name}.")
        return {}

    yearly_data = (
        df_with_orgs.group_by(["year", "org"])
        .agg(pl.col("author_id").n_unique().alias("unique_authors_per_org"))
        .with_columns(
            share=pl.col("unique_authors_per_org")
            / pl.col("unique_authors_per_org").sum().over("year")
        )
        .group_by("year")
        .agg(
            raw_hhi=(pl.col("share") ** 2).sum(),
            n_orgs=pl.col("org").n_unique(),
        )
    )

    year_independence = {}
    for row in yearly_data.iter_rows(named=True):
        yr = int(row["year"])
        raw_hhi = float(row["raw_hhi"])
        n = int(row["n_orgs"])

        if n <= 1:
            score = 0.0
        else:
            hhi_star = (raw_hhi - (1.0 / n)) / (1.0 - (1.0 / n))
            score = 1.0 - hhi_star

        year_independence[yr] = {"Independence": max(0.0, min(1.0, float(score)))}

    return year_independence


def compute_pluralism_author_gini_per_year(
    ctx: IndividualProjectContext,
) -> Dict[int, Dict[str, float]]:
    """Computes Pluralism Score as the inverse Gini Coefficient of proposal authorship."""
    first_revision = ctx.proposal_revisions.group_by("proposal_id").agg(
        min_rev=pl.col("revision_index").min(),
        created_at=pl.col("created_at").min(),
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
        logging.warning(f"No valid authorships found in {ctx.project_name}.")
        return {}

    gini_per_year = (
        df.group_by(["year", "author_id"])
        .agg(pl.len().alias("contribution_count"))
        .group_by("year")
        .agg((1.0 - _polars_gini_expr("contribution_count")).alias("inverse_gini"))
    )

    return {int(r[0]): {"Pluralism": float(r[1])} for r in gini_per_year.iter_rows()}


def compute_representation_comment_gini_per_year(
    ctx: IndividualProjectContext,
) -> Dict[int, Dict[str, float]]:
    """Computes Representation Score as the inverse Gini Coefficient of comments."""
    df = ctx.comments.filter(
        pl.col("author_id").is_not_null() & pl.col("created_at").is_not_null()
    ).with_columns(pl.col("created_at").dt.year().alias("year"))

    if df.is_empty():
        logging.warning(f"No valid comments found in {ctx.project_name}.")
        return {}

    gini_per_year = (
        df.group_by(["year", "author_id"])
        .agg(pl.len().alias("comment_count"))
        .group_by("year")
        .agg((1.0 - _polars_gini_expr("comment_count")).alias("inverse_gini"))
    )

    return {
        int(r[0]): {"Representation": float(r[1])} for r in gini_per_year.iter_rows()
    }


def compute_centralization_metrics_per_year(
    ctx: IndividualProjectContext,
) -> Dict[int, Dict[str, float]]:
    """Computes Network Decentralization Scores (1 - Centralization) for structural elements."""
    rev_df = (
        ctx.proposal_revision_authors.join(
            ctx.proposal_revisions, on=["proposal_id", "revision_index"], how="inner"
        )
        .with_columns(
            year=pl.col("created_at").dt.year(),
            group_id=pl.col("proposal_id"),
        )
        .select(["year", "group_id", "author_id"])
    )

    comment_df = (
        ctx.comments.filter(
            pl.col("proposal_id").is_not_null()
            & pl.col("author_id").is_not_null()
            & pl.col("created_at").is_not_null()
        )
        .with_columns(
            year=pl.col("created_at").dt.year(),
            group_id=pl.col("proposal_id"),
        )
        .select(["year", "group_id", "author_id"])
    )

    combined_df = pl.concat([rev_df, comment_df]).filter(
        pl.col("year").is_not_null() & pl.col("author_id").is_not_null()
    )
    if combined_df.is_empty():
        return {}

    year_groups = combined_df.group_by(["year", "group_id"]).agg(
        pl.col("author_id").unique().alias("members")
    )

    edges_map = {}
    year_nodes = {}

    for row in year_groups.iter_rows(named=True):
        yr = int(row["year"])
        members = sorted(list(row["members"]))
        if len(members) < 2:
            continue

        if yr not in year_nodes:
            year_nodes[yr] = set()
        year_nodes[yr].update(members)

        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                u, v = members[i], members[j]
                edge_key = (yr, u, v)
                edges_map[edge_key] = edges_map.get(edge_key, 0) + 1

    year_centralization = {}
    all_years = sorted(list(year_nodes.keys()))

    for yr in all_years:
        nodes = list(year_nodes[yr])
        n = len(nodes)

        if n < 3:
            year_centralization[yr] = {
                "Degree": 1.0,
                "Betweenness": 1.0,
                "Closeness": 1.0,
            }
            continue

        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        g = rx.PyGraph()
        g.add_nodes_from(nodes)

        yr_edges = [
            (node_to_idx[k[1]], node_to_idx[k[2]], 1.0 / weight)
            for k, weight in edges_map.items()
            if k[0] == yr
        ]

        if not yr_edges:
            year_centralization[yr] = {
                "Degree": 1.0,
                "Betweenness": 1.0,
                "Closeness": 1.0,
            }
            continue

        g.add_edges_from(yr_edges)

        # Degree
        deg_map = {node_idx: g.degree(node_idx) for node_idx in range(n)}
        deg_vals = [deg / (n - 1) for deg in deg_map.values()]
        max_deg = max(deg_vals) if deg_vals else 0.0
        deg_sum_diff = sum((max_deg - v) for v in deg_vals)
        deg_denom = float(n - 2)
        deg_index = float(deg_sum_diff / deg_denom if deg_denom > 0 else 0.0)

        # Betweenness
        weighted_cb = {i: 0.0 for i in range(n)}
        path_res = rx.all_pairs_dijkstra_shortest_paths(
            g, edge_cost_fn=lambda e: float(e)
        )

        total_paths = 0
        for s, targets in path_res.items():
            for t, path in targets.items():
                if s == t:
                    continue
                total_paths += 1
                if len(path) > 2:
                    inner_nodes = path[1:-1]
                    for node in inner_nodes:
                        weighted_cb[node] += 1.0

        cb_vals = [
            score / total_paths if total_paths > 0 else 0.0
            for score in weighted_cb.values()
        ]
        max_cb = max(cb_vals) if cb_vals else 0.0
        cb_sum_diff = sum((max_cb - v) for v in cb_vals)
        cb_denom = float((n - 1) * (n - 2))
        cb_index = float((cb_sum_diff / cb_denom) * n if cb_denom > 0 else 0.0)

        # Closeness
        cc_map = rx.closeness_centrality(g)
        cc_vals = list(cc_map.values())
        max_cc = max(cc_vals) if cc_vals else 0.0
        cc_sum_diff = sum((max_cc - v) for v in cc_vals)
        cc_denom = float(((n - 1) * (n - 2)) / (2 * n - 3))
        cc_index = float(cc_sum_diff / cc_denom if cc_denom > 0 else 0.0)

        year_centralization[yr] = {
            "Degree": float(max(0.0, min(1.0, 1.0 - deg_index))),
            "Betweenness": float(max(0.0, min(1.0, 1.0 - cb_index))),
            "Closeness": float(max(0.0, min(1.0, 1.0 - cc_index))),
        }

    return year_centralization


def compute_newcomers_onboarding_per_year(
    ctx: IndividualProjectContext,
) -> Dict[int, Dict[str, float]]:
    """Computes Autonomous Participation Score as the proportion of newcomer proposals."""
    global_earliest = (
        ctx.proposal_revision_authors.join(
            ctx.proposal_revisions, on=["proposal_id", "revision_index"], how="inner"
        )
        .group_by("author_id")
        .agg(first_global_activity=pl.col("created_at").min())
    )

    first_revision = ctx.proposal_revisions.group_by("proposal_id").agg(
        min_rev=pl.col("revision_index").min(),
        proposal_created_at=pl.col("created_at").min(),
    )

    proposals_df = (
        first_revision.join(
            ctx.proposal_revision_authors,
            left_on=["proposal_id", "min_rev"],
            right_on=["proposal_id", "revision_index"],
            how="inner",
        )
        .join(global_earliest, on="author_id", how="left")
        .with_columns(year=pl.col("proposal_created_at").dt.year())
        .filter(pl.col("year").is_not_null())
    )
    if proposals_df.is_empty():
        return {}

    onboarding_rate = (
        proposals_df.with_columns(
            author_is_experienced=pl.col("first_global_activity")
            < pl.col("proposal_created_at")
        )
        .group_by(["year", "proposal_id"])
        .agg(has_experienced_author=pl.col("author_is_experienced").any())
        .with_columns(is_newcomer_proposal=~pl.col("has_experienced_author"))
        .group_by("year")
        .agg(
            total_proposals=pl.col("proposal_id").n_unique(),
            newcomer_proposals=pl.col("is_newcomer_proposal").sum(),
        )
        .with_columns(rate=pl.col("newcomer_proposals") / pl.col("total_proposals"))
        .sort("year")
    )
    return {
        int(row["year"]): {"Autonomous Participation": float(row["rate"])}
        for row in onboarding_rate.iter_rows(named=True)
    }


# =====================================================================
# Grouping Visualization Helpers
# =====================================================================


def _extract_grouped_metric_labels(
    projects_stats: List[GovernanceProjectStats],
) -> List[Tuple[str, str]]:
    """Identifies and resolves flat mapping of (Parent Governance Stat, Nested Statistic Name) tuples."""
    pairs = set()
    for p_entry in projects_stats:
        for stat_name, item in p_entry.items.items():
            for metric_key in item.metrics.keys():
                pairs.add((stat_name, metric_key))
    return sorted(list(pairs), key=lambda x: (x[0], x[1]))


def plot_consolidated_line_charts(
    projects_stats: List[GovernanceProjectStats], output_dir: Path
) -> None:
    """Plots line metrics dynamically grouping metrics under their respective Governance domains."""
    grouped_pairs = _extract_grouped_metric_labels(projects_stats)
    colors = plt.colormaps["tab10"](np.linspace(0, 1, max(10, len(projects_stats))))

    for stat_name, metric_key in grouped_pairs:
        fig, ax = plt.subplots(figsize=(11, 6))
        has_data = False

        for idx, p_stat in enumerate(projects_stats):
            if (
                stat_name in p_stat.items
                and metric_key in p_stat.items[stat_name].metrics
            ):
                timeline = p_stat.items[stat_name].metrics[metric_key]
                if not timeline:
                    continue

                has_data = True
                years = sorted(timeline.keys())
                scores = [timeline[y] for y in years]

                ax.plot(
                    years,
                    scores,
                    marker="o",
                    linewidth=2,
                    markersize=6,
                    color=colors[idx % len(colors)],
                    label=p_stat.project_name,
                )

        display_title = (
            f"{stat_name} $\\rightarrow$ {metric_key}"
            if stat_name != metric_key
            else stat_name
        )
        ax.set_title(
            f"Trend Analysis: {display_title}", fontsize=12, fontweight="bold", pad=15
        )
        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel("Score Profile Index", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if has_data:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                borderaxespad=0,
                frameon=True,
            )

        plt.tight_layout()
        safe_filename = "".join(
            c if c.isalnum() else "_" for c in f"{stat_name}_{metric_key}"
        ).lower()
        plt.savefig(
            str(output_dir / f"line_{safe_filename}.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()


def plot_combined_heatmap(
    projects_stats: List[GovernanceProjectStats], output_dir: Path
) -> None:
    """Generates a composite cross-project overview grouping sub-statistics transparently."""
    if not projects_stats:
        return

    grouped_pairs = _extract_grouped_metric_labels(projects_stats)
    unique_projects = sorted([p.project_name for p in projects_stats])

    if not grouped_pairs or not unique_projects:
        return

    matrix_data = np.zeros((len(grouped_pairs), len(unique_projects)))
    project_map = {p.project_name: p for p in projects_stats}

    for m_idx, (stat_name, metric_key) in enumerate(grouped_pairs):
        for p_idx, p_name in enumerate(unique_projects):
            p_obj = project_map[p_name]
            if (
                stat_name in p_obj.items
                and metric_key in p_obj.items[stat_name].metrics
            ):
                vals = list(p_obj.items[stat_name].metrics[metric_key].values())
                matrix_data[m_idx, p_idx] = np.mean(vals) if vals else 0.0

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(matrix_data, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)

    row_labels = [
        f"{s_name} ({m_key})" if s_name != m_key else s_name
        for s_name, m_key in grouped_pairs
    ]

    ax.set_xticks(np.arange(len(unique_projects)))
    ax.set_yticks(np.arange(len(grouped_pairs)))
    ax.set_xticklabels(unique_projects, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)

    for i in range(len(grouped_pairs)):
        for j in range(len(unique_projects)):
            ax.text(
                j,
                i,
                f"{matrix_data[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if matrix_data[i, j] < 0.7 else "white",
                fontweight="bold",
            )

    fig.colorbar(im, ax=ax, label="Normalized Mean Performance Value")
    ax.set_title(
        "Grouped Governance Profile Cross-Heatmap Summary",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    plt.savefig(
        str(output_dir / "combined_grouped_heatmap.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_combined_parallel_coordinates(
    projects_stats: List[GovernanceProjectStats], output_dir: Path
) -> None:
    """Constructs a Parallel Coordinates Plot visualizing project trajectory paths."""
    if not projects_stats:
        return

    grouped_pairs = _extract_grouped_metric_labels(projects_stats)
    if len(grouped_pairs) < 2:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    x_positions = np.arange(len(grouped_pairs))

    colors = plt.colormaps["Set1"](np.linspace(0, 1, len(projects_stats)))
    project_map = {p.project_name: p for p in projects_stats}

    for idx, (p_name, p_obj) in enumerate(project_map.items()):
        trajectory_vector = []
        for stat_name, metric_key in grouped_pairs:
            if (
                stat_name in p_obj.items
                and metric_key in p_obj.items[stat_name].metrics
            ):
                vals = list(p_obj.items[stat_name].metrics[metric_key].values())
                trajectory_vector.append(np.mean(vals) if vals else 0.0)
            else:
                trajectory_vector.append(0.0)

        ax.plot(
            x_positions,
            trajectory_vector,
            marker="s",
            linewidth=2.5,
            markersize=8,
            color=colors[idx],
            label=p_name,
            alpha=0.85,
        )

    for pos in x_positions:
        ax.axvline(pos, color="black", linestyle="-", alpha=0.25, zorder=1)

    axis_labels = [
        f"{s_name}\n({m_key})" if s_name != m_key else s_name
        for s_name, m_key in grouped_pairs
    ]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(axis_labels, fontsize=9, fontweight="bold")
    ax.set_ylabel("Metric Evaluation Scores", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), title="Evaluated Projects")
    ax.set_title(
        "Grouped Governance Parallel Coordinates Structural Vector",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(
        str(output_dir / "combined_grouped_parallel_coordinates.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_project_radars(
    projects_stats: List[GovernanceProjectStats], output_dir: Path
) -> None:
    """Generates proportional nested radar charts.

    Splits total polar space uniformly by amount of unique Governance Stats,
    then subdivides each allocation block by its respective nested metrics list,
    leaving visual margin gaps between parent groupings.
    """
    radar_dir = output_dir / "radar_charts"
    radar_dir.mkdir(exist_ok=True)

    # Resolve parent stats layout tracking lists
    grouped_pairs = _extract_grouped_metric_labels(projects_stats)
    if not grouped_pairs:
        return

    parent_stats = sorted(list({pair[0] for pair in grouped_pairs}))
    num_parents = len(parent_stats)

    # Setup structural margins (allocating 15 degrees per gap boundary spacing)
    gap_margin_radians = np.radians(15.0)
    total_gap_allowance = gap_margin_radians * num_parents
    remaining_pool_arc = (2 * np.pi) - total_gap_allowance

    # Share arc evenly across core items blocks
    arc_per_parent = remaining_pool_arc / num_parents

    # 1. Map angular coordinates dynamically
    angles_list = []
    labels_list = []

    current_cursor_angle = 0.0

    for p_stat in parent_stats:
        child_metrics = [pair[1] for pair in grouped_pairs if pair[0] == p_stat]
        num_children = len(child_metrics)

        # Subdivide parent arc slice equally across inner statistics metrics count
        sub_arc_step = arc_per_parent / max(1, num_children)

        for idx, m_key in enumerate(child_metrics):
            # Place coordinate in center of child's sub-slice assignment
            target_angle = (
                current_cursor_angle + (idx * sub_arc_step) + (sub_arc_step / 2.0)
            )
            angles_list.append(target_angle)

            label_text = f"{p_stat}\n({m_key})" if p_stat != m_key else p_stat
            labels_list.append(label_text)

        current_cursor_angle += arc_per_parent + gap_margin_radians

    # Close polar geometric render arrays seamlessly
    closed_angles = angles_list + [angles_list[0]]

    # 2. Render charts per project
    for p_obj in projects_stats:
        values = []
        for stat_name, metric_key in grouped_pairs:
            if (
                stat_name in p_obj.items
                and metric_key in p_obj.items[stat_name].metrics
            ):
                vals = list(p_obj.items[stat_name].metrics[metric_key].values())
                values.append(np.mean(vals) if vals else 0.0)
            else:
                values.append(0.0)
        closed_values = values + [values[0]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        ax.set_xticks(angles_list)
        ax.set_xticklabels(labels_list, color="#2c3e50", size=9, weight="bold")
        ax.set_ylim(0, 1.0)

        # Adjust label margins and text bounding collisions
        ax.tick_params(axis="x", pad=22)

        # Plot structural trace path mappings
        ax.plot(
            closed_angles,
            closed_values,
            color="#1f77b4",
            linewidth=2,
            linestyle="solid",
        )
        ax.fill(closed_angles, closed_values, color="#1f77b4", alpha=0.18)

        # Grid system aesthetics lines
        ax.set_rlabel_position(0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=9)

        ax.set_title(
            f"{p_obj.project_name} - Proportional Governance Profile",
            fontsize=13,
            fontweight="bold",
            pad=30,
        )

        safe_name = "".join(
            c if c.isalnum() else "_" for c in p_obj.project_name
        ).lower()
        plt.savefig(
            str(radar_dir / f"proportional_radar_{safe_name}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


# =====================================================================
# Central Analytical Orchestration Pipeline Engine
# =====================================================================


def show_governance_statistics(
    projects: List[IndividualProjectContext], output_dir: Path
) -> None:
    """Calculates all metrics exactly once, writing visualizations and summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Definitive Computation Context Mapping Pipeline
    computation_registry = {
        "Independence": compute_independence_hhi_per_year,
        "Pluralism": compute_pluralism_author_gini_per_year,
        "Representation": compute_representation_comment_gini_per_year,
        "Decentralization Decision-Making": compute_centralization_metrics_per_year,
        "Autonomous Participation": compute_newcomers_onboarding_per_year,
    }

    # 1. Pipeline Calculation Phase
    project_records: List[GovernanceProjectStats] = []

    for ctx in projects:
        print(f"Analyzing structure variables for project: {ctx.project_name}")
        project_container = GovernanceProjectStats(project_name=ctx.project_name)

        for stat_domain, compute_fn in computation_registry.items():
            try:
                yearly_results = compute_fn(ctx)
                if yearly_results:
                    stat_item = project_container.get_or_create_item(stat_domain)
                    for year, metrics_dict in yearly_results.items():
                        for metric_name, score in metrics_dict.items():
                            if metric_name not in stat_item.metrics:
                                stat_item.metrics[metric_name] = {}
                            stat_item.metrics[metric_name][year] = score
            except Exception as ex:
                logging.error(
                    f"Failed pipeline calculations for domain {stat_domain} on {ctx.project_name}: {ex}"
                )

        project_records.append(project_container)

    # 2. Rendering Plotting Execution Phase
    print("Executing visualization generation tasks over grouped structural records...")
    plot_consolidated_line_charts(project_records, output_dir)
    plot_combined_heatmap(project_records, output_dir)
    plot_combined_parallel_coordinates(project_records, output_dir)
    plot_project_radars(project_records, output_dir)

    # 3. Output Aggregations Data Frame Tables Phase
    grouped_pairs = _extract_grouped_metric_labels(project_records)
    table_headers = [f"{s} ({m})" if s != m else s for s, m in grouped_pairs]
    table_data = {"Governance Domain Framework Structure": table_headers}

    for p_record in project_records:
        means = []
        for stat_name, metric_key in grouped_pairs:
            if (
                stat_name in p_record.items
                and metric_key in p_record.items[stat_name].metrics
            ):
                vals = list(p_record.items[stat_name].metrics[metric_key].values())
                means.append(round(np.mean(vals), 4) if vals else 0.0)
            else:
                means.append(0.0)
        table_data[p_record.project_name] = means

    summary_df = pl.DataFrame(table_data)

    with pl.Config(
        tbl_formatting="markdown", tbl_hide_dataframe_shape=True, tbl_rows=-1
    ):
        (output_dir / "governance_statistics.md").write_text(
            str(summary_df), encoding="utf-8"
        )

    print(
        f"Data calculations pipeline terminated successfully. Files written to: {output_dir}"
    )
