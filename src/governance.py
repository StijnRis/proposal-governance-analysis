import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import rustworkx as rx  # Vastly faster Rust alternative to NetworkX
from matplotlib.ticker import MaxNLocator

from src.statistics import IndividualProjectContext


@dataclass
class GovMetricConfig:
    """Configuration mapping for processing, tracking, and plotting governance dynamics."""

    filename: str
    title: str
    description: str
    compute_fn: Callable[[IndividualProjectContext], Dict[int, float]]


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
# Metric Transformation Engines (Optimized Slicing Context)
# =====================================================================


def compute_independence_hhi_per_year(
    ctx: IndividualProjectContext,
) -> Dict[int, float]:
    """Computes Normalized Inverse HHI for organizational independence per year."""
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

        year_independence[yr] = max(0.0, min(1.0, float(score)))

    return year_independence


def compute_pluralism_author_gini_per_year(
    ctx: IndividualProjectContext,
) -> Dict[int, float]:
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

    return {int(r[0]): float(r[1]) for r in gini_per_year.iter_rows()}


def compute_representation_comment_gini_per_year(
    ctx: IndividualProjectContext,
) -> Dict[int, float]:
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

    return {int(r[0]): float(r[1]) for r in gini_per_year.iter_rows()}


def compute_centralization_metrics_per_year(
    ctx: IndividualProjectContext,
) -> Dict[int, Dict[str, float]]:
    """
    Computes Network Decentralization Scores (1 - Centralization)
    for Degree, Betweenness, and Closeness natively optimized via rustworkx.

    Returns:
        Dict[year, {"degree": float, "betweenness": float, "closeness": float}]
    """
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

    # Gather interactions cleanly without full Cartesian matrix self-joins
    year_groups = combined_df.group_by(["year", "group_id"]).agg(
        pl.col("author_id").unique().alias("members")
    )

    # Reconstruct edges safely in Python to pass straight to PyO3 Rust layers
    edges_map = {}  # key: (year, u, v) -> weight
    year_nodes = {}  # key: year -> set of nodes

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

        # Freeman centralization breaks down mathematically with fewer than 3 nodes
        if n < 3:
            year_centralization[yr] = {
                "degree": 1.0,
                "betweenness": 1.0,
                "closeness": 1.0,
            }
            continue

        # Map nodes to indices for rustworkx compatibility
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        # Construct Rust-backed Graph
        g = rx.PyGraph()
        g.add_nodes_from(nodes)

        yr_edges = [
            (node_to_idx[k[1]], node_to_idx[k[2]], 1.0 / weight)
            for k, weight in edges_map.items()
            if k[0] == yr
        ]

        if not yr_edges:
            year_centralization[yr] = {
                "degree": 1.0,
                "betweenness": 1.0,
                "closeness": 1.0,
            }
            continue

        g.add_edges_from(yr_edges)

        # --- 1. DEGREE CENTRALIZATION ---
        # rustworkx degree returns raw integer degrees; normalize manually by dividing by (n - 1)
        deg_map = {node_idx: g.degree(node_idx) for node_idx in range(n)}
        deg_vals = [deg / (n - 1) for deg in deg_map.values()]
        max_deg = max(deg_vals) if deg_vals else 0.0
        deg_sum_diff = sum((max_deg - v) for v in deg_vals)
        deg_denom = float(
            n - 2
        )  # Freeman denominator for normalized scores: (n-1)(n-2)/(n-1) -> (n-2)
        deg_index = float(deg_sum_diff / deg_denom if deg_denom > 0 else 0.0)

        # --- 2. BETWEENNESS CENTRALIZATION ---
        # Normalized by default in rustworkx
        cb_map = rx.betweenness_centrality(g, normalized=True)
        cb_vals = list(cb_map.values())
        max_cb = max(cb_vals) if cb_vals else 0.0
        cb_sum_diff = sum((max_cb - v) for v in cb_vals)
        cb_denom = float((n - 1) * (n - 2))
        cb_index = float(cb_sum_diff / cb_denom if cb_denom > 0 else 0.0)

        # --- 3. CLOSENESS CENTRALIZATION ---
        # rustworkx closeness calculates the standard normalized metric out of the box
        cc_map = rx.closeness_centrality(g)
        cc_vals = list(cc_map.values())
        max_cc = max(cc_vals) if cc_vals else 0.0
        cc_sum_diff = sum((max_cc - v) for v in cc_vals)
        cc_denom = float(((n - 1) * (n - 2)) / (2 * n - 3))
        cc_index = float(cc_sum_diff / cc_denom if cc_denom > 0 else 0.0)

        # --- CONVERT CENTRALIZATION INDEX -> DECENTRALIZATION SCORE ---
        # Keeping your existing project logic: Decentralization = 1.0 - Centralization Index
        year_centralization[yr] = {
            "degree": float(max(0.0, min(1.0, 1.0 - deg_index))),
            "betweenness": float(max(0.0, min(1.0, 1.0 - cb_index))),
            "closeness": float(max(0.0, min(1.0, 1.0 - cc_index))),
        }

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
    return dict(
        zip(onboarding_rate["year"].to_list(), onboarding_rate["rate"].to_list())
    )


GOV_REGISTRY = [
    GovMetricConfig(
        "independence.png",
        "Independence",
        "Inverse HHI",
        compute_independence_hhi_per_year,
    ),
    GovMetricConfig(
        "pluralism.png",
        "Pluralism",
        "Inverse Gini Author Variety",
        compute_pluralism_author_gini_per_year,
    ),
    GovMetricConfig(
        "representation.png",
        "Representation",
        "Inverse Gini Comment Concentration",
        compute_representation_comment_gini_per_year,
    ),
    GovMetricConfig(
        "decentralized_decision_making.png",
        "Decentralized Decision-Making",
        "Inverse Betweenness",
        compute_betweenness_centralization_per_year,
    ),
    GovMetricConfig(
        "autonomous_participation.png",
        "Autonomous Participation",
        "Onboarding Rate",
        compute_newcomers_onboarding_per_year,
    ),
]

# =====================================================================
# UI Engine Rendering Core Mechanics
# =====================================================================


def _render_consolidated_gov_plot(
    contexts: List[IndividualProjectContext],
    cfg: GovMetricConfig,
    computed_cache: Dict[str, Dict[str, Dict[int, float]]],
    output_dir: Path,
) -> None:
    """Plots lines for all projects into a single unified figure."""
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.colormaps["tab10"](np.linspace(0, 1, max(10, len(contexts))))

    has_data = False
    for idx, ctx in enumerate(contexts):
        scores_dict = computed_cache[ctx.project_name][cfg.title]
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

    ax.set_title(
        cfg.title + f" ({cfg.description})", fontsize=12, fontweight="bold", pad=15
    )
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(str(output_dir / cfg.filename), dpi=300, bbox_inches="tight")
    plt.close()


def _get_radar_data_cached(
    project_scores: Dict[str, Dict[int, float]],
    target_years: set[int],
) -> Tuple[List[str], List[float]]:
    """Generates radar axis information using the computed metrics cache."""
    labels = []
    values = []

    for cfg in GOV_REGISTRY:
        labels.append(cfg.title)
        scores_dict = project_scores[cfg.title]
        filtered_scores = [v for k, v in scores_dict.items() if k in target_years]
        values.append(float(np.mean(filtered_scores)) if filtered_scores else 0.0)

    return labels, values


def _render_project_radar(
    project_name: str, labels: List[str], values: List[float], output_dir: Path
) -> None:
    """Generates and saves a single radar chart for one project."""
    num_vars = len(labels)
    angles = (
        np.linspace(0, 2 * np.pi, num_vars, endpoint=False) + (np.pi / len(labels))
    ).tolist()
    angles += angles[:1]
    closed_values = values + values[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    ax.set_ylim(0, 1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Configure Outer X-Axis Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="#4a4a4a", size=16)
    ax.tick_params(axis="x", pad=5)

    # Dynamic alignment loop for outer labels
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        angle_deg = np.degrees(angle)
        if 0 < angle_deg < 180:
            label.set_horizontalalignment("left")
        elif 180 < angle_deg < 360:
            label.set_horizontalalignment("right")
        else:
            label.set_horizontalalignment("center")
        label.set_wrap(True)

    # Configure Inner Y-Axis Labels
    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=14)

    # Plot data
    ax.plot(angles, closed_values, color="#1f77b4", linewidth=2, linestyle="solid")
    ax.fill(angles, closed_values, color="#1f77b4", alpha=0.25)

    # Title
    ax.set_title(
        f"{project_name} Governance Profile", fontsize=18, fontweight="bold", pad=30
    )

    # Save logic
    safe_name = "".join(c if c.isalnum() else "_" for c in project_name).lower()
    plt.tight_layout()
    plt.savefig(
        str(output_dir / f"radar_{safe_name}.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def show_governance_statistics(
    projects: List[IndividualProjectContext], output_dir: Path
) -> None:
    """Calculates all metrics exactly once, writing visualizations and summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)

    computed_cache: Dict[str, Dict[str, Dict[int, float]]] = {}
    for ctx in projects:
        print(f"Computing governance metrics for project: {ctx.project_name}")
        computed_cache[ctx.project_name] = {
            cfg.title: cfg.compute_fn(ctx) for cfg in GOV_REGISTRY
        }

    for cfg in GOV_REGISTRY:
        _render_consolidated_gov_plot(projects, cfg, computed_cache, output_dir)

    current_year = datetime.datetime.now().year
    target_years = set(range(current_year - 4, current_year + 1))

    table_data = {"Governance Metric": [cfg.title for cfg in GOV_REGISTRY]}

    radar_output_dir = output_dir / "radar_charts"
    radar_output_dir.mkdir(exist_ok=True)
    for ctx in projects:
        labels, values = _get_radar_data_cached(
            computed_cache[ctx.project_name], target_years
        )
        _render_project_radar(ctx.project_name, labels, values, radar_output_dir)
        table_data[ctx.project_name] = [round(v, 4) for v in values]

    summary_df = pl.DataFrame(table_data)

    # Save as Markdown Table directly via Polars options string output
    with pl.Config(
        tbl_formatting="markdown", tbl_hide_dataframe_shape=True, tbl_rows=-1
    ):
        (output_dir / "governance_statistics.md").write_text(
            str(summary_df), encoding="utf-8"
        )

    # Clean programmatic structural creation for LaTeX without pandas dependency
    latex_lines = [
        r"\begin{tabular}{l" + "c" * len(projects) + "}",
        r"\hline",
        " & ".join(summary_df.columns) + r" \\",
        r"\hline",
    ]
    for row in summary_df.iter_rows():
        latex_lines.append(" & ".join(str(x) for x in row) + r" \\")
    latex_lines.extend([r"\hline", r"\end{tabular}"])

    (output_dir / "governance_statistics.tex").write_text(
        "\n".join(latex_lines), encoding="utf-8"
    )
