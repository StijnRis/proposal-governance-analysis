# =====================================================================
# Structured Data Containers
# =====================================================================
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import polars as pl
import rustworkx as rx
import scipy.stats as stats

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
    group_data: Dict[
        str, Dict[str, np.ndarray]
    ]  # Formatted as: {dimension: {"Community": array, "Corporate": array}}


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
    """
    Computes Inverse HHI (1 - HHI) for organizational independence.
    Optimized: Isolates core proposal authorship shares without log-comment smoothing.
    """
    # Group strictly by proposal authorship to separate organizational control
    proposal_counts = ctx.proposal_revision_authors.group_by(["author_id"]).agg(
        pl.len().alias("proposal_count")
    )

    df_authors = (
        ctx.affiliations.join(ctx.organisations, on="organisation_id", how="inner")
        .rename({"person_id": "author_id"})
        .join(
            proposal_counts, on="author_id", how="inner"
        )  # Inner join focuses on governing actors
    )

    first_proposal_date = ctx.proposal_revisions.group_by("proposal_id").agg(
        pl.col("created_at").min()
    )

    df = (
        first_proposal_date.join(
            ctx.proposal_revision_authors, on="proposal_id", how="inner"
        )
        .with_columns(pl.col("created_at").dt.year().alias("year"))
        .join(df_authors, on="author_id", how="inner")
    )

    if df.height < 2:
        return 0.0 if mode == "most_recent" else {}

    def _calculate_core_hhi(sliced_df: pl.DataFrame) -> float:
        if sliced_df.height < 2:
            return 0.0

        # Calculate HHI on absolute proposal distribution weight per organization
        org_shares = sliced_df.group_by("organisation_name").agg(
            pl.col("proposal_count").sum().alias("total_org_weight")
        )

        total_weight = org_shares["total_org_weight"].sum()
        if total_weight == 0:
            return 0.0

        shares = org_shares["total_org_weight"] / total_weight
        hhi = (shares**2).sum()
        return float(1.0 - hhi)

    max_year = df.select(pl.col("year").max()).item()

    if mode == "most_recent":
        start_year = max_year - window_size + 1
        pooled_df = df.filter(pl.col("year").is_between(start_year, max_year))
        return _calculate_core_hhi(pooled_df)
    elif mode == "all_windows":
        min_year = df.select(pl.col("year").min()).item()
        results = {}
        for end_year in range(min_year, max_year + 1):
            start_year = end_year - window_size + 1
            window_df = df.filter(pl.col("year").is_between(start_year, end_year))
            results[end_year] = _calculate_core_hhi(window_df)
        return results
    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_pluralism_author_gini(
    ctx: IndividualProjectContext, window_size: int, mode: str
) -> Union[float, Dict[int, float]]:
    """
    Computes Pluralism Score as the inverse Gini Coefficient of proposal authorship.
    Optimized: Measures all historical revision interactions rather than isolating index 0.
    """
    # Evaluate across all revisions to track collaborative community code adjustments
    df = (
        ctx.proposal_revisions.join(
            ctx.proposal_revision_authors,
            on=["proposal_id", "revision_index"],
            how="inner",
        )
        .with_columns(pl.col("created_at").dt.year().alias("year"))
        .filter(pl.col("year").is_not_null())
    )
    if df.is_empty():
        return 0.0 if mode == "most_recent" else {}

    def _calculate_core_pluralism(sliced_df: pl.DataFrame) -> float:
        if sliced_df.is_empty():
            return 0.0
        gini_expr = (
            sliced_df.group_by("author_id")
            .agg(pl.len().alias("contribution_count"))
            .select(
                (1.0 - _polars_gini_expr("contribution_count")).alias("inverse_gini")
            )
        )
        return float(gini_expr.item()) if not gini_expr.is_empty() else 0.0

    max_year = df.select(pl.col("year").max()).item()

    if mode == "most_recent":
        start_year = max_year - window_size + 1
        pooled_df = df.filter(pl.col("year").is_between(start_year, max_year))
        return _calculate_core_pluralism(pooled_df)
    elif mode == "all_windows":
        min_year = df.select(pl.col("year").min()).item()
        results = {}
        for end_year in range(min_year, max_year + 1):
            start_year = end_year - window_size + 1
            window_df = df.filter(pl.col("year").is_between(start_year, end_year))
            results[end_year] = _calculate_core_pluralism(window_df)
        return results
    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_centralization_metrics(
    ctx: IndividualProjectContext, window_size: int, mode: str
) -> Union[float, Dict[int, float]]:
    """
    Computes Network Betweenness Decentralization Index using bipartite participation.
    Optimized: Removes intra-project self-normalization loops to expose variance.
    """
    rev_df = (
        ctx.proposal_revision_authors.join(
            ctx.proposal_revisions, on=["proposal_id", "revision_index"], how="inner"
        )
        .with_columns(
            pl.lit(1.0).alias("auth_weight"),
            year=pl.col("created_at").dt.year(),
        )
        .select(["year", "proposal_id", "author_id", "auth_weight"])
    )

    comment_df = (
        ctx.comments.filter(
            pl.col("proposal_id").is_not_null() & pl.col("author_id").is_not_null()
        )
        .group_by(["proposal_id", "author_id"])
        .agg(pl.len().alias("c_ip"))
        # Absolute mathematical scaling without internal division filters
        .with_columns(normalized_comment_weight=pl.col("c_ip").add(1).log())
    )

    combined_participation = (
        rev_df.join(
            comment_df.select(
                ["proposal_id", "author_id", "normalized_comment_weight"]
            ),
            on=["proposal_id", "author_id"],
            how="outer",
        )
        .with_columns(
            pl.col("auth_weight").fill_null(0.0),
            pl.col("normalized_comment_weight").fill_null(0.0),
        )
        .with_columns(
            (pl.col("auth_weight") + pl.col("normalized_comment_weight")).alias(
                "edge_weight"
            )
        )
    )
    if combined_participation.is_empty():
        return 0.0 if mode == "most_recent" else {}

    def _calculate_core_centralization(sliced_df: pl.DataFrame) -> float:
        if sliced_df.is_empty():
            return 1.0

        year_groups = sliced_df.group_by("proposal_id").agg(
            pl.col("author_id").unique().alias("members")
        )

        edges_map = {}
        nodes_set = set()

        for row in year_groups.iter_rows(named=True):
            members = sorted(list(row["members"]))
            if len(members) < 2:
                continue

            nodes_set.update(members)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    edge_key = (members[i], members[j])
                    edges_map[edge_key] = edges_map.get(edge_key, 0) + 1

        nodes = list(nodes_set)
        n = len(nodes)
        if n < 3:
            return 1.0

        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        g = rx.PyGraph()
        g.add_nodes_from(nodes)

        yr_edges = [
            (node_to_idx[k[0]], node_to_idx[k[1]], 1.0 / w)
            for k, w in edges_map.items()
        ]

        if not yr_edges:
            return 1.0

        g.add_edges_from(yr_edges)
        cb_dict = dict(rx.graph_betweenness_centrality(g, normalized=True))

        cb_vals = [cb_dict.get(i, 0.0) for i in range(n)]
        max_cb = max(cb_vals) if cb_vals else 0.0
        cb_sum_diff = sum((max_cb - v) for v in cb_vals)

        cb_denom = float(n - 1)
        cb_index = float(cb_sum_diff / cb_denom if cb_denom > 0 else 0.0)
        return float(max(0.0, min(1.0, 1.0 - cb_index)))

    max_year = combined_participation.select(pl.col("year").max()).item()

    if mode == "most_recent":
        start_year = max_year - window_size + 1
        pooled_df = combined_participation.filter(
            pl.col("year").is_between(start_year, max_year)
        )
        return _calculate_core_centralization(pooled_df)
    elif mode == "all_windows":
        min_year = combined_participation.select(pl.col("year").min()).item()
        results = {}
        for end_year in range(min_year, max_year + 1):
            start_year = end_year - window_size + 1
            window_df = combined_participation.filter(
                pl.col("year").is_between(start_year, end_year)
            )
            results[end_year] = _calculate_core_centralization(window_df)
        return results
    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_newcomers_onboarding(
    ctx: IndividualProjectContext, window_size: int, mode: str
) -> Union[float, Dict[int, float]]:
    """Computes Autonomous Participation Score as the proportion of first-time

    contributors among all active contributors within configurable windows.
    """
    author_first_activity = (
        ctx.proposal_revision_authors.join(
            ctx.proposal_revisions, on=["proposal_id", "revision_index"], how="inner"
        )
        .group_by("author_id")
        .agg(first_activity_year=pl.col("created_at").dt.year().min())
    )

    all_activity = (
        ctx.proposal_revision_authors.join(
            ctx.proposal_revisions, on=["proposal_id", "revision_index"], how="inner"
        )
        .with_columns(year=pl.col("created_at").dt.year())
        .select(["year", "author_id"])
        .unique()
    )

    if all_activity.is_empty():
        return 0.0 if mode == "most_recent" else {}

    def _calculate_core_onboarding(sliced_df: pl.DataFrame) -> float:
        if sliced_df.is_empty():
            return 0.0
        total_active_authors = sliced_df.select("author_id").n_unique()
        if total_active_authors == 0:
            return 0.0

        joined = sliced_df.join(author_first_activity, on="author_id", how="inner")
        window_min_year = sliced_df.select(pl.col("year").min()).item()
        window_max_year = sliced_df.select(pl.col("year").max()).item()

        new_authors = (
            joined.filter(
                pl.col("first_activity_year").is_between(
                    window_min_year, window_max_year
                )
            )
            .select("author_id")
            .n_unique()
        )
        return float(new_authors / total_active_authors)

    max_year = all_activity.select(pl.col("year").max()).item()

    if mode == "most_recent":
        start_year = max_year - window_size + 1
        pooled_df = all_activity.filter(pl.col("year").is_between(start_year, max_year))
        return _calculate_core_onboarding(pooled_df)

    elif mode == "all_windows":
        min_year = all_activity.select(pl.col("year").min()).item()
        results = {}
        for end_year in range(min_year, max_year + 1):
            start_year = end_year - window_size + 1
            window_df = all_activity.filter(
                pl.col("year").is_between(start_year, end_year)
            )
            results[end_year] = _calculate_core_onboarding(window_df)
        return results
    else:
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
        start_year = max_year - window_size + 1
        pooled_df = df.filter(pl.col("year").is_between(start_year, max_year))
        return _calculate_core_representation(pooled_df)

    elif mode == "all_windows":
        min_year = df.select(pl.col("year").min()).item()
        results = {}
        for end_year in range(min_year, max_year + 1):
            start_year = end_year - window_size + 1
            window_df = df.filter(pl.col("year").is_between(start_year, end_year))
            results[end_year] = _calculate_core_representation(window_df)
        return results
    else:
        raise ValueError(f"Unknown mode: {mode}")


def calculate_known_groups_validity(
    project_records: List[GovernanceProjectStats],
    ordered_keys: List[str],
) -> KnownGroupsValidationResult:
    """
    Groups metrics into archetypes and executes statistical validity tests.
    Returns a structured data container holding zero plotting dependencies.
    """
    group_mapping = {
        "JavaScript": "Community",
        "Kubernetes": "Community",
        "NumPy": "Community",
        "Pandas": "Community",
        "Python": "Community",
        "Rust": "Community",
        "Kotlin": "Corporate",
        "OpenJDK": "Corporate",
        "Swift": "Corporate",
        "C++": "Corporate",
    }

    # Extract active dimensions present across records
    present_dims = {dim for p in project_records for dim in p.metrics}
    dimensions = [dim for dim in ordered_keys if dim in present_dims]

    records_list = []
    for p_obj in project_records:
        row = {
            "Project": p_obj.project_name,
            "Group": group_mapping.get(p_obj.project_name, "Unknown"),
        }
        for dim in dimensions:
            row[dim] = p_obj.pooled_metrics.get(dim, 0.0)
        records_list.append(row)

    df_all = pl.DataFrame(records_list)
    df_test = df_all.filter(pl.col("Group").is_in(["Community", "Corporate"]))

    validity_rows = []
    group_data = {}

    for dim in dimensions:
        comm_vals = (
            df_test.filter(pl.col("Group") == "Community")
            .select(dim)
            .to_series()
            .to_numpy()
        )
        corp_vals = (
            df_test.filter(pl.col("Group") == "Corporate")
            .select(dim)
            .to_series()
            .to_numpy()
        )

        # Store arrays for downstream plotting
        group_data[dim] = {"Community": comm_vals, "Corporate": corp_vals}

        mean_comm, mean_corp = np.mean(comm_vals), np.mean(corp_vals)
        median_comm, median_corp = np.median(comm_vals), np.median(corp_vals)

        # Better small-sample Mann-Whitney calculation
        if len(comm_vals) > 0 and len(corp_vals) > 0:
            u_stat, p_val = stats.mannwhitneyu(
                comm_vals, corp_vals, alternative="two-sided", method="exact"
            )

            std_comm, std_corp = np.std(comm_vals, ddof=1), np.std(corp_vals, ddof=1)
            denom = len(comm_vals) + len(corp_vals) - 2
            pooled_std = (
                np.sqrt(
                    (
                        ((len(comm_vals) - 1) * std_comm**2)
                        + ((len(corp_vals) - 1) * std_corp**2)
                    )
                    / denom
                )
                if denom > 0
                else 0.0
            )
            cohen_d = (mean_comm - mean_corp) / pooled_std if pooled_std != 0 else 0.0
        else:
            p_val = 1.0
            cohen_d = 0.0

        # Tiered verification threshold for Small-N constraints
        if p_val < 0.05:
            is_valid = "Yes (p < 0.05)"
        elif abs(cohen_d) >= 0.8:
            is_valid = f"Practical (d = {round(cohen_d, 2)})"
        else:
            is_valid = "No"

        validity_rows.append(
            {
                "Governance Dimension": dim,
                "Corporate Mean": round(mean_corp, 4),
                "Corporate Median": round(median_corp, 4),
                "Community Mean": round(mean_comm, 4),
                "Community Median": round(median_comm, 4),
                "p-value": round(p_val, 4),
                "Cohen's d": round(cohen_d, 4),
                "Discriminant Validity Status": is_valid,
            }
        )

    return KnownGroupsValidationResult(
        ordered_keys=ordered_keys,
        dimensions=dimensions,
        validity_rows=validity_rows,
        group_data=group_data,
    )


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
            # 1. Timeline Breakdown: window_size=1 across historical slices maps exactly to separate annual datapoints
            yearly_results = compute_fn(ctx, window_size=1, mode="all_windows")
            if yearly_results:
                project_container.metrics[stat_domain] = yearly_results

            # 2. Pooled Matrix: window_size=5 tracks a block snapshot for heatmaps, parallel trends, and tabular indices
            pooled_result = compute_fn(ctx, window_size=5, mode="most_recent")
            project_container.pooled_metrics[stat_domain] = pooled_result

        project_records.append(project_container)

    known_groups_result = calculate_known_groups_validity(project_records, ordered_keys)

    return project_records, ordered_keys, known_groups_result
