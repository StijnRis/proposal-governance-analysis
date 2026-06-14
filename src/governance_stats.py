# =====================================================================
# Structured Data Containers
# =====================================================================
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import polars as pl
import rustworkx as rx

from statistics2 import IndividualProjectContext


@dataclass
class GovernanceProjectStats:
    """Stores both timeline history and pooled multi-year summary metrics for a project."""

    project_name: str
    metrics: Dict[str, Dict[int, float]] = field(default_factory=dict)
    pooled_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class KnownGroupsValidationResult:
    """Holds structured statistical test outputs and arrays cleanly divorced from plotting layers."""

    ordered_keys: List[str]
    dimensions: List[str]
    validity_rows: List[Dict[str, Any]]
    group_data: Dict[str, Dict[str, np.ndarray]]


# =====================================================================
# Atomic Mathematical Helpers
# =====================================================================


def _polars_gini_expr(col_name: str) -> pl.Expr:
    """Returns a Polars expression calculating an unbiased Gini Coefficient."""
    valid_sorted = pl.col(col_name).drop_nulls().sort()
    n = valid_sorted.len()
    sum_x = valid_sorted.sum()

    index = valid_sorted.rank("ordinal")
    gini = (2 * (index * valid_sorted).sum() / (n * sum_x)) - ((n + 1) / n)

    return pl.when((n <= 1) | (sum_x == 0)).then(0.0).otherwise(gini)


# =====================================================================
# Metric Transformation Engines (Configurable Sliding Window Designs)
# =====================================================================


def compute_independence_hhi(
    ctx: IndividualProjectContext, window_size: int, mode: str
) -> Union[float, Dict[int, float]]:
    """Computes Inverse HHI (1 - HHI) for organizational independence.

    Formula: w_a = p_a + log(1 + c_a)
    Unmapped/missing affiliations are excluded to avoid data skew.
    """
    proposal_counts = (
        ctx.proposal_revisions.join(
            ctx.proposal_revision_authors,
            on=["proposal_id", "revision_index"],
            how="inner",
        )
        .with_columns(pl.col("created_at").dt.year().alias("year"))
        .filter(pl.col("year").is_not_null())
        .group_by(["year", "author_id"])
        .agg(pl.col("proposal_id").n_unique().alias("p_a"))
    )

    comment_counts = (
        ctx.comments.filter(
            pl.col("author_id").is_not_null() & pl.col("created_at").is_not_null()
        )
        .with_columns(pl.col("created_at").dt.year().alias("year"))
        .group_by(["year", "author_id"])
        .agg(pl.len().alias("c_a"))
    )

    all_years_authors = pl.concat(
        [
            proposal_counts.select(["year", "author_id"]),
            comment_counts.select(["year", "author_id"]),
        ]
    ).unique()

    if all_years_authors.is_empty():
        return 0.0 if mode == "most_recent" else {}

    author_metrics_stream = (
        all_years_authors.join(proposal_counts, on=["year", "author_id"], how="left")
        .join(comment_counts, on=["year", "author_id"], how="left")
        .with_columns(pl.col("p_a").fill_null(0), pl.col("c_a").fill_null(0))
        .with_columns((pl.col("p_a") + (pl.col("c_a") + 1).log()).alias("w_a"))
        .join(
            ctx.affiliations.rename({"person_id": "author_id"}),
            on="author_id",
            how="left",
        )
        .join(ctx.organisations, on="organisation_id", how="left")
        .select(["year", "author_id", "organisation_name", "w_a"])
    )

    def _calculate_core_hhi(sliced_df: pl.DataFrame) -> float:
        valid_affiliations = sliced_df.filter(pl.col("organisation_name").is_not_null())
        if valid_affiliations.is_empty():
            return 0.0

        org_shares = valid_affiliations.group_by("organisation_name").agg(
            pl.col("w_a").sum().alias("total_org_weight")
        )

        total_weight = org_shares["total_org_weight"].sum()
        if total_weight == 0:
            return 0.0

        return float(1.0 - ((org_shares["total_org_weight"] / total_weight) ** 2).sum())

    max_year = author_metrics_stream.select(pl.col("year").max()).item()

    if mode == "most_recent":
        pooled_df = author_metrics_stream.filter(
            pl.col("year").is_between(max_year - window_size + 1, max_year)
        )
        return _calculate_core_hhi(pooled_df)

    if mode == "all_windows":
        min_year = author_metrics_stream.select(pl.col("year").min()).item()
        return {
            end_year: _calculate_core_hhi(
                author_metrics_stream.filter(
                    pl.col("year").is_between(end_year - window_size + 1, end_year)
                )
            )
            for end_year in range(min_year, max_year + 1)
        }

    raise ValueError(f"Unknown mode: {mode}")


def compute_pluralism_author_gini(
    ctx: IndividualProjectContext, window_size: int, mode: str
) -> Union[float, Dict[int, float]]:
    """Computes Pluralism Score as the inverse Gini Coefficient of distinct proposal authorship."""
    proposal_years = ctx.proposal_revisions.select(
        ["proposal_id", pl.col("created_at").dt.year().alias("year")]
    ).unique()

    author_proposals = ctx.proposal_revision_authors.select(
        ["proposal_id", "author_id"]
    ).unique()

    df = proposal_years.join(author_proposals, on="proposal_id", how="inner").filter(
        pl.col("year").is_not_null()
    )

    if df.is_empty():
        return 0.0 if mode == "most_recent" else {}

    def _calculate_core_pluralism(sliced_df: pl.DataFrame) -> float:
        if sliced_df.is_empty():
            return 0.0
        gini_expr = (
            sliced_df.group_by("author_id")
            .agg(pl.col("proposal_id").n_unique().alias("x_i"))
            .select((1.0 - _polars_gini_expr("x_i")).alias("inverse_gini"))
        )
        return float(gini_expr.item()) if not gini_expr.is_empty() else 0.0

    max_year = df.select(pl.col("year").max()).item()

    if mode == "most_recent":
        pooled_df = df.filter(
            pl.col("year").is_between(max_year - window_size + 1, max_year)
        )
        return _calculate_core_pluralism(pooled_df)

    if mode == "all_windows":
        min_year = df.select(pl.col("year").min()).item()
        return {
            end_year: _calculate_core_pluralism(
                df.filter(
                    pl.col("year").is_between(end_year - window_size + 1, end_year)
                )
            )
            for end_year in range(min_year, max_year + 1)
        }

    raise ValueError(f"Unknown mode: {mode}")


def compute_centralization_metrics(
    ctx: IndividualProjectContext, window_size: int, mode: str
) -> Union[float, Dict[int, float]]:
    """Computes Network Betweenness Decentralization Index using a true bipartite network."""
    rev_events = ctx.proposal_revision_authors.join(
        ctx.proposal_revisions,
        on=["proposal_id", "revision_index"],
        how="inner",
    ).select(
        [
            pl.col("created_at").dt.year().alias("year"),
            "proposal_id",
            "author_id",
            pl.lit(1.0).alias("a_ip"),
            pl.lit(0.0).alias("c_ip_count"),
        ]
    )

    comment_events = ctx.comments.filter(
        pl.col("proposal_id").is_not_null() & pl.col("author_id").is_not_null()
    ).select(
        [
            pl.col("created_at").dt.year().alias("year"),
            "proposal_id",
            "author_id",
            pl.lit(0.0).alias("a_ip"),
            pl.lit(1.0).alias("c_ip_count"),
        ]
    )

    all_events = pl.concat([rev_events, comment_events])

    if all_events.is_empty():
        return 0.0 if mode == "most_recent" else {}

    def _calculate_core_centralization(sliced_df: pl.DataFrame) -> float:
        if sliced_df.is_empty():
            return 1.0

        raw_participation = sliced_df.group_by(["proposal_id", "author_id"]).agg(
            [
                pl.col("a_ip").max().alias("a_ip"),
                pl.col("c_ip_count").sum().alias("c_ip"),
            ]
        )

        m_expr = raw_participation.select(
            (pl.col("c_ip") + 1).log().max().alias("max_m")
        )
        m = (
            float(m_expr.item())
            if not m_expr.is_empty() and m_expr.item() is not None
            else 0.0
        )

        participation = raw_participation.with_columns(
            w_ip=pl.col("a_ip") + ((pl.col("c_ip") + 1).log() / m)
            if m > 0
            else pl.col("a_ip")
        )

        unique_authors = participation["author_id"].unique().to_list()
        unique_proposals = participation["proposal_id"].unique().to_list()
        n_people = len(unique_authors)

        if n_people < 3:
            return 1.0

        g = rx.PyGraph()
        node_map = {}

        for author in unique_authors:
            node_map[("person", author)] = g.add_node(f"person_{author}")
        for prop in unique_proposals:
            node_map[("proposal", prop)] = g.add_node(f"proposal_{prop}")

        edges_to_add = []
        for row in participation.iter_rows(named=True):
            p_node = node_map[("person", row["author_id"])]
            prop_node = node_map[("proposal", row["proposal_id"])]
            weight = float(row["w_ip"])
            edges_to_add.append(
                (p_node, prop_node, 1.0 / weight if weight > 0 else float("inf"))
            )

        g.add_edges_from(edges_to_add)
        cb_dict = dict(rx.graph_betweenness_centrality(g, normalized=True))

        person_indices = [node_map[("person", a)] for a in unique_authors]
        cb_vals = [cb_dict.get(idx, 0.0) for idx in person_indices]

        max_cb = max(cb_vals) if cb_vals else 0.0
        cb_sum_diff = sum((max_cb - v) for v in cb_vals)

        cb_denom = float(n_people - 1)
        cb_index = float(cb_sum_diff / cb_denom if cb_denom > 0 else 0.0)

        return float(max(0.0, min(1.0, 1.0 - cb_index)))

    max_year = all_events.select(pl.col("year").max()).item()

    if mode == "most_recent":
        window_df = all_events.filter(
            pl.col("year").is_between(max_year - window_size + 1, max_year)
        )
        return _calculate_core_centralization(window_df)

    if mode == "all_windows":
        min_year = all_events.select(pl.col("year").min()).item()
        return {
            end_year: _calculate_core_centralization(
                all_events.filter(
                    pl.col("year").is_between(end_year - window_size + 1, end_year)
                )
            )
            for end_year in range(min_year, max_year + 1)
        }

    raise ValueError(f"Unknown mode: {mode}")


def compute_newcomers_onboarding(
    ctx: IndividualProjectContext, window_size: int, mode: str
) -> Union[float, Dict[int, float]]:
    """Computes Autonomous Participation Score using Proposal-Weighted Fractional Authorship."""
    author_first_activity = (
        ctx.proposal_revision_authors.join(
            ctx.proposal_revisions, on=["proposal_id", "revision_index"], how="inner"
        )
        .sort("created_at")
        .group_by("author_id")
        .agg(
            first_proposal_id=pl.col("proposal_id").first(),
            first_activity_year=pl.col("created_at").dt.year().first(),
        )
    )

    proposal_base = (
        ctx.proposal_revision_authors.join(
            ctx.proposal_revisions, on=["proposal_id", "revision_index"], how="inner"
        )
        .with_columns(year=pl.col("created_at").dt.year())
        .select(["year", "proposal_id", "author_id"])
        .unique()
    )

    if proposal_base.is_empty():
        return 0.0 if mode == "most_recent" else {}

    def _calculate_core_onboarding(sliced_df: pl.DataFrame) -> float:
        if sliced_df.is_empty():
            return 0.0

        window_min_year = sliced_df.select(pl.col("year").min()).item()
        window_max_year = sliced_df.select(pl.col("year").max()).item()

        processed = sliced_df.join(
            author_first_activity, on="author_id", how="inner"
        ).with_columns(
            pl.when(
                (pl.col("proposal_id") == pl.col("first_proposal_id"))
                & pl.col("first_activity_year").is_between(
                    window_min_year, window_max_year
                )
            )
            .then(1.0)
            .otherwise(0.0)
            .alias("is_newcomer")
        )

        proposal_shares = (
            processed.group_by("proposal_id")
            .agg(total_authors=pl.len(), newcomer_authors=pl.col("is_newcomer").sum())
            .with_columns(
                fractional_newcomer_share=pl.col("newcomer_authors")
                / pl.col("total_authors")
            )
        )

        total_proposals = proposal_shares.height
        if total_proposals == 0:
            return 0.0

        return float(
            proposal_shares["fractional_newcomer_share"].sum() / total_proposals
        )

    max_year = proposal_base.select(pl.col("year").max()).item()

    if mode == "most_recent":
        pooled_df = proposal_base.filter(
            pl.col("year").is_between(max_year - window_size + 1, max_year)
        )
        return _calculate_core_onboarding(pooled_df)

    if mode == "all_windows":
        min_year = proposal_base.select(pl.col("year").min()).item()
        return {
            end_year: _calculate_core_onboarding(
                proposal_base.filter(
                    pl.col("year").is_between(end_year - window_size + 1, end_year)
                )
            )
            for end_year in range(min_year, max_year + 1)
        }

    raise ValueError(f"Unknown mode: {mode}")


def compute_representation_comment_gini(
    ctx: IndividualProjectContext, window_size: int, mode: str
) -> Union[float, Dict[int, float]]:
    """Computes Representation Score as the inverse Gini Coefficient of comments over windows."""
    df = ctx.comments.filter(
        pl.col("author_id").is_not_null() & pl.col("created_at").is_not_null()
    ).with_columns(pl.col("created_at").dt.year().alias("year"))

    if df.is_empty():
        return 0.0 if mode == "most_recent" else {}

    def _calculate_core_representation(sliced_df: pl.DataFrame) -> float:
        if sliced_df.is_empty():
            return 0.0
        gini_expr = (
            sliced_df.group_by("author_id")
            .agg(pl.len().alias("comment_count"))
            .select((1.0 - _polars_gini_expr("comment_count")).alias("inverse_gini"))
        )
        return float(gini_expr.item()) if not gini_expr.is_empty() else 0.0

    max_year = df.select(pl.col("year").max()).item()

    if mode == "most_recent":
        pooled_df = df.filter(
            pl.col("year").is_between(max_year - window_size + 1, max_year)
        )
        return _calculate_core_representation(pooled_df)

    if mode == "all_windows":
        min_year = df.select(pl.col("year").min()).item()
        return {
            end_year: _calculate_core_representation(
                df.filter(
                    pl.col("year").is_between(end_year - window_size + 1, end_year)
                )
            )
            for end_year in range(min_year, max_year + 1)
        }

    raise ValueError(f"Unknown mode: {mode}")


def get_governance_statistics(
    projects: List[IndividualProjectContext],
) -> Tuple[List[GovernanceProjectStats], List[str], KnownGroupsValidationResult]:
    """Calculates flat trend lines and robust multi-year pooled profiles over all data assets."""
    computation_registry = {
        "Independence": compute_independence_hhi,
        "Pluralism": compute_pluralism_author_gini,
        "Representation": compute_representation_comment_gini,
        "Decentralized Decision-Making": compute_centralization_metrics,
        "Autonomous Participation": compute_newcomers_onboarding,
    }

    ordered_keys = list(computation_registry.keys())
    project_records: List[GovernanceProjectStats] = []

    for ctx in projects:
        print(f"Analyzing structure variables for project: {ctx.project_name}")
        project_container = GovernanceProjectStats(project_name=ctx.project_name)

        for stat_domain, compute_fn in computation_registry.items():
            yearly_results = compute_fn(ctx, window_size=1, mode="all_windows")
            if yearly_results:
                project_container.metrics[stat_domain] = yearly_results

            project_container.pooled_metrics[stat_domain] = compute_fn(
                ctx, window_size=5, mode="most_recent"
            )

        project_records.append(project_container)

    return (
        project_records,
        ordered_keys,
        calculate_known_groups_validity(project_records, ordered_keys),
    )
