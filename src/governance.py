"""Governance metrics computation, aggregation, and consolidated single-plot visualization."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import networkx as nx
import numpy as np
import polars as pl
from matplotlib import pyplot as plt

# Reusing the project context slicing framework built in your previous file
from src.statistics import IndividualProjectContext


@dataclass
class GovMetricConfig:
    """Configuration mapping for processing, tracking, and plotting governance dynamics."""

    key: str
    filename: str
    title: str
    compute_fn: Callable[[IndividualProjectContext], Dict[int, float]]


# =====================================================================
# Atomic Mathematical Helpers
# =====================================================================


def _polars_gini(values: pl.Series) -> float:
    """Computes an unbiased Gini Coefficient for an array of values."""
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
    """Computes Herfindahl-Hirschman Index (HHI) for organizational independence per year."""
    first_proposal_date = ctx.proposal_revisions.group_by("proposal_id").agg(
        pl.col("created_at").min()
    )

    df = (
        first_proposal_date.join(ctx.proposal_revision_authors, on="proposal_id", how="inner")
        .with_columns(pl.col("created_at").dt.year().alias("year"))
        .filter(pl.col("year").is_not_null())
        .select(["year", "proposal_id", "author_id"])
    )
    if df.is_empty():
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

    yearly_hhi = (
        df_with_orgs.group_by(["year", "org"])
        .agg(pl.col("author_id").n_unique().alias("unique_authors_per_org"))
        .with_columns(
            (
                pl.col("unique_authors_per_org")
                / pl.col("unique_authors_per_org").sum().over("year")
            ).alias("share")
        )
        .group_by("year")
        .agg((pl.col("share") ** 2).sum().alias("hhi"))
        .sort("year")
    )
    return dict(zip(yearly_hhi["year"].to_list(), yearly_hhi["hhi"].to_list()))


def compute_pluralism_author_gini_per_year(
    ctx: IndividualProjectContext,
) -> Dict[int, float]:
    """Computes Gini Coefficient of proposal creation distribution among authors per year."""
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
        return {}

    author_counts = df.group_by(["year", "author_id"]).agg(
        pl.len().alias("contribution_count")
    )
    return {
        int(year[0]): _polars_gini(group["contribution_count"])
        for year, group in author_counts.partition_by("year", as_dict=True).items()
    }


def compute_representation_comment_gini_per_year(
    ctx: IndividualProjectContext,
) -> Dict[int, float]:
    """Computes Gini Coefficient of comment volume distributions among authors per year."""
    df = ctx.comments.filter(
        pl.col("author_id").is_not_null() & pl.col("created_at").is_not_null()
    ).with_columns(pl.col("created_at").dt.year().alias("year"))

    if df.is_empty():
        return {}

    comment_counts = df.group_by(["year", "author_id"]).agg(
        pl.len().alias("comment_count")
    )
    return {
        int(year[0]): _polars_gini(group["comment_count"])
        for year, group in comment_counts.partition_by("year", as_dict=True).items()
    }


def compute_betweenness_centralization_per_year(
    ctx: IndividualProjectContext,
) -> Dict[int, float]:
    """Computes social network betweenness centralization using Polars graph-edge preparation."""
    rev_df = (
        ctx.proposal_revision_authors.join(
            ctx.proposal_revisions, on=["proposal_id", "revision_index"], how="inner"
        )
        .with_columns(
            [
                pl.col("created_at").dt.year().alias("year"),
                (
                    pl.col("proposal_id")
                    + pl.lit("_")
                    + pl.col("revision_index").cast(pl.String)
                ).alias("group_id"),
            ]
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
            [
                pl.col("created_at").dt.year().alias("year"),
                pl.col("proposal_id").alias("group_id"),
            ]
        )
        .select(["year", "group_id", "author_id"])
    )

    combined_df = pl.concat([rev_df, comment_df]).filter(pl.col("year").is_not_null())
    if combined_df.is_empty():
        return {}

    year_centralization = {}
    for year, group in combined_df.partition_by("year", as_dict=True).items():
        nodes = set(group["author_id"].drop_nulls().to_list())
        interactions = (
            group.group_by("group_id")
            .agg(pl.col("author_id").unique().alias("authors"))
            .filter(pl.col("authors").list.len() >= 2)
        )

        weights = {}
        for row in interactions.iter_rows(named=True):
            authors = sorted(row["authors"])
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    key = (authors[i], authors[j])
                    weights[key] = weights.get(key, 0) + 1

        if len(nodes) < 3 or not weights:
            year_centralization[int(year[0])] = 0.0
            continue

        G = nx.Graph()
        G.add_nodes_from(list(nodes))
        for (a, b), w in weights.items():
            G.add_edge(a, b, weight=(1.0 / w))

        centrality = nx.betweenness_centrality(G, weight="weight", normalized=True)
        if not centrality:
            year_centralization[int(year[0])] = 0.0
            continue

        max_cb = max(centrality.values())
        n = G.number_of_nodes()
        sum_diff = sum((max_cb - v) for v in centrality.values())
        denom = (n - 1) * (n - 2)
        year_centralization[int(year[0])] = float(
            sum_diff / denom if denom > 0 else 0.0
        )

    return year_centralization


def compute_newcomers_onboarding_per_year(
    ctx: IndividualProjectContext,
) -> Dict[int, float]:
    """Computes ratio of new-contributor-driven proposals submitted per year."""
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


# Registry containing all governance calculation rules
GOV_REGISTRY = [
    GovMetricConfig(
        "independence_hhi",
        "affiliation_diversity_hhi.png",
        "Affiliation Diversity: Herfindahl-Hirschman Index (HHI)",
        compute_independence_hhi_per_year,
    ),
    GovMetricConfig(
        "pluralism_gini",
        "author_variety_gini.png",
        "Author Variety: Gini Coefficient",
        compute_pluralism_author_gini_per_year,
    ),
    GovMetricConfig(
        "representation_gini",
        "comment_concentration_gini.png",
        "Comment Concentration: Gini Coefficient",
        compute_representation_comment_gini_per_year,
    ),
    GovMetricConfig(
        "betweenness_centralization",
        "path_control_centralization.png",
        "Path Control: Betweenness Centralization",
        compute_betweenness_centralization_per_year,
    ),
    GovMetricConfig(
        "newcomer_success_rate",
        "newcomer_onboarding_rate.png",
        "Newcomer Onboarding Rate",
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
    
    # Generate distinct colors automatically based on the number of projects
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(contexts))))
    
    has_data = False
    for idx, ctx in enumerate(contexts):
        scores_dict = cfg.compute_fn(ctx)
        if not scores_dict:
            continue
            
        has_data = True
        years = sorted(scores_dict.keys())
        scores = [scores_dict[y] for y in years]
        
        # Plot timeline line per project matching its designated index color
        ax.plot(
            years, 
            scores, 
            marker="o", 
            linewidth=2, 
            markersize=6, 
            color=colors[idx % len(colors)], 
            label=ctx.project_name
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
        # Include legend only if valid lines have been parsed and drawn
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=True)

    ax.set_title(cfg.title, fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(str(output_dir / cfg.filename), dpi=300, bbox_inches="tight")
    plt.close()


# =====================================================================
# Unified Coordination Entrypoints
# =====================================================================


def show_governance_statistics(
    project: list[IndividualProjectContext], output_dir: Path
) -> None:
    """Calculates all governance indices, producing consolidated multi-line plots directly into the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each strategic metric configuration tracking performance globally
    for cfg in GOV_REGISTRY:
        _render_consolidated_gov_plot(project, cfg, output_dir)