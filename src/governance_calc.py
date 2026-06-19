import os
from dataclasses import dataclass, field
from typing import Dict, List, Union

import numpy as np
import polars as pl
import rustworkx as rx

from statistics2 import IndividualProjectContext

# =====================================================================
# Explicitly Typed Metric Objects
# =====================================================================


@dataclass(frozen=True)
class MetricInterval:
    """Holds a point estimate alongside its empirical confidence bounds."""

    val: float
    ci_low: float
    ci_high: float


@dataclass
class MetricTimeline:
    """Manages the time-series profile for a single structural dimension."""

    # Maps year -> Interval data structure
    windows: Dict[int, MetricInterval] = field(default_factory=dict)


@dataclass
class ProjectMetricsProfile:
    """Groups all 5 distinct metric timelines to prevent arbitrary string dictionary lookups."""

    independence: MetricTimeline = field(default_factory=MetricTimeline)
    independence_no_discard: MetricTimeline = field(default_factory=MetricTimeline)
    pluralism: MetricTimeline = field(default_factory=MetricTimeline)
    representation: MetricTimeline = field(default_factory=MetricTimeline)
    decentralization: MetricTimeline = field(default_factory=MetricTimeline)
    autonomous_participation: MetricTimeline = field(default_factory=MetricTimeline)


@dataclass
class ProjectPooledProfile:
    """Groups the multi-year pooled summary metric intervals."""

    independence: MetricInterval = field(
        default_factory=lambda: MetricInterval(0.0, 0.0, 0.0)
    )
    independence_no_discard: MetricInterval = field(
        default_factory=lambda: MetricInterval(0.0, 0.0, 0.0)
    )
    pluralism: MetricInterval = field(
        default_factory=lambda: MetricInterval(0.0, 0.0, 0.0)
    )
    representation: MetricInterval = field(
        default_factory=lambda: MetricInterval(0.0, 0.0, 0.0)
    )
    decentralization: MetricInterval = field(
        default_factory=lambda: MetricInterval(0.0, 0.0, 0.0)
    )
    autonomous_participation: MetricInterval = field(
        default_factory=lambda: MetricInterval(0.0, 0.0, 0.0)
    )


# =====================================================================
# Refactored Primary Container
# =====================================================================


@dataclass
class GovernanceProjectStats:
    """Stores explicit timeline histories and multi-year summary metrics for a project."""

    project_name: str
    release_year: int
    metrics: ProjectMetricsProfile = field(default_factory=ProjectMetricsProfile)
    pooled_metrics: ProjectPooledProfile = field(default_factory=ProjectPooledProfile)


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


def _bootstrap_interval(
    df: pl.DataFrame,
    calc_fn,
    cluster_col: str,
    n_bootstraps: int,
    ci: float = 0.95,
) -> Dict[str, float]:
    """Helper that handles the resampling and percentile extraction."""
    point_est = calc_fn(df)
    if df.is_empty():
        return {"val": point_est, "ci_low": point_est, "ci_high": point_est}

    # Extract unique cluster keys to sample with replacement safely
    unique_keys = df[cluster_col].unique().to_numpy()
    n_keys = len(unique_keys)

    if n_keys < 5:  # Too few observations to calculate a meaningful variance
        return {"val": point_est, "ci_low": point_est, "ci_high": point_est}

    boot_estimates = []
    # Using a deterministic seed sequence for reproducible CIs
    rng = np.random.default_rng(42)

    for _ in range(n_bootstraps):
        sample_keys = rng.choice(unique_keys, size=n_keys, replace=True)
        # Filter dataframe to match the resampled clusters
        boot_df = df.filter(pl.col(cluster_col).is_in(sample_keys))
        boot_estimates.append(calc_fn(boot_df))

    low_p = (1.0 - ci) / 2.0
    high_p = 1.0 - low_p

    return {
        "val": float(point_est),
        "ci_low": float(np.percentile(boot_estimates, low_p * 100)),
        "ci_high": float(np.percentile(boot_estimates, high_p * 100)),
    }


# =====================================================================
# Metric Transformation Engines with Bootstrap Support
# =====================================================================


def compute_independence_hhi_do_no_discard_and_no_bootstrapping(
    ctx: IndividualProjectContext,
    window_size: int,
    mode: str,
    n_bootstraps: int,
) -> Union[Dict[str, float], Dict[int, Dict[str, float]]]:
    """Computes Inverse HHI with 95% Bootstrap Confidence Intervals without discarding unaffiliated authors."""
    return compute_independence_hhi(
        ctx, window_size, mode, 1, discard_unaffiliated=False
    )


def compute_independence_hhi(
    ctx: IndividualProjectContext,
    window_size: int,
    mode: str,
    n_bootstraps: int,
    discard_unaffiliated: bool = True,
) -> Union[Dict[str, float], Dict[int, Dict[str, float]]]:
    """Computes Inverse HHI with 95% Bootstrap Confidence Intervals."""
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
        return (
            {"val": 0.0, "ci_low": 0.0, "ci_high": 0.0} if mode == "most_recent" else {}
        )

    # Base stream setup
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
    )

    # Process affiliation logic based on the boolean flag
    if discard_unaffiliated:
        author_metrics_stream = author_metrics_stream.filter(
            pl.col("organisation_name").is_not_null()
        ).select(["year", "author_id", "organisation_name", "w_a"])
    else:
        author_metrics_stream = author_metrics_stream.with_columns(
            pl.when(pl.col("organisation_name").is_null())
            .then(pl.lit("independent_") + pl.col("author_id").cast(pl.Utf8))
            .otherwise(pl.col("organisation_name"))
            .alias("organisation_name")
        ).select(["year", "author_id", "organisation_name", "w_a"])

    def _score_fn(sliced_df: pl.DataFrame) -> float:
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

    def _calculate_core_hhi(sliced_df: pl.DataFrame) -> Dict[str, float]:
        return _bootstrap_interval(
            sliced_df,
            _score_fn,
            cluster_col="author_id",
            n_bootstraps=n_bootstraps,
        )

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
    ctx: IndividualProjectContext, window_size: int, mode: str, n_bootstraps: int
) -> Union[Dict[str, float], Dict[int, Dict[str, float]]]:
    """Computes Pluralism Score with 95% Bootstrap Confidence Intervals."""
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
        return (
            {"val": 0.0, "ci_low": 0.0, "ci_high": 0.0} if mode == "most_recent" else {}
        )

    def _score_fn(sliced_df: pl.DataFrame) -> float:
        if sliced_df.is_empty():
            return 0.0
        gini_expr = (
            sliced_df.group_by("author_id")
            .agg(pl.col("proposal_id").n_unique().alias("x_i"))
            .select((1.0 - _polars_gini_expr("x_i")).alias("inverse_gini"))
        )
        return float(gini_expr.item()) if not gini_expr.is_empty() else 0.0

    def _calculate_core_pluralism(sliced_df: pl.DataFrame) -> Dict[str, float]:
        return _bootstrap_interval(
            sliced_df, _score_fn, cluster_col="author_id", n_bootstraps=n_bootstraps
        )

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
    ctx: IndividualProjectContext,
    window_size: int,
    mode: str,
    n_bootstraps: int,
) -> Union[Dict[str, float], Dict[int, Dict[str, float]]]:
    """Computes Network Betweenness Centralization with 95% Bootstrap Confidence Intervals."""
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
        return (
            {"val": 0.0, "ci_low": 0.0, "ci_high": 0.0} if mode == "most_recent" else {}
        )

    def _score_fn(sliced_df: pl.DataFrame) -> float:
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

    def _calculate_core_centralization(sliced_df: pl.DataFrame) -> Dict[str, float]:
        # Resampling grouped by proposal structure ensures network paths hold integrity
        return _bootstrap_interval(
            sliced_df, _score_fn, cluster_col="proposal_id", n_bootstraps=n_bootstraps
        )

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
    ctx: IndividualProjectContext, window_size: int, mode: str, n_bootstraps: int
) -> Union[Dict[str, float], Dict[int, Dict[str, float]]]:
    """Computes Autonomous Participation Score with 95% Bootstrap Confidence Intervals."""
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
        return (
            {"val": 0.0, "ci_low": 0.0, "ci_high": 0.0} if mode == "most_recent" else {}
        )

    def _score_fn(sliced_df: pl.DataFrame) -> float:
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
        return (
            float(proposal_shares["fractional_newcomer_share"].sum() / total_proposals)
            if total_proposals > 0
            else 0.0
        )

    def _calculate_core_onboarding(sliced_df: pl.DataFrame) -> Dict[str, float]:
        return _bootstrap_interval(
            sliced_df, _score_fn, cluster_col="proposal_id", n_bootstraps=n_bootstraps
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
    ctx: IndividualProjectContext, window_size: int, mode: str, n_bootstraps: int
) -> Union[Dict[str, float], Dict[int, Dict[str, float]]]:
    """Computes Representation Score with 95% Bootstrap Confidence Intervals."""
    df = ctx.comments.filter(
        pl.col("author_id").is_not_null() & pl.col("created_at").is_not_null()
    ).with_columns(pl.col("created_at").dt.year().alias("year"))

    if df.is_empty():
        return (
            {"val": 0.0, "ci_low": 0.0, "ci_high": 0.0} if mode == "most_recent" else {}
        )

    def _score_fn(sliced_df: pl.DataFrame) -> float:
        if sliced_df.is_empty():
            return 0.0
        gini_expr = (
            sliced_df.group_by("author_id")
            .agg(pl.len().alias("comment_count"))
            .select((1.0 - _polars_gini_expr("comment_count")).alias("inverse_gini"))
        )
        return float(gini_expr.item()) if not gini_expr.is_empty() else 0.0

    def _calculate_core_representation(sliced_df: pl.DataFrame) -> Dict[str, float]:
        return _bootstrap_interval(
            sliced_df, _score_fn, cluster_col="author_id", n_bootstraps=n_bootstraps
        )

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


# =====================================================================
# Main Orchestration Pipeline Execution
# =====================================================================
RELEASE_YEARS = {
    "C++": 1985,
    "JavaScript": 1995,
    "Kotlin": 2016,
    "Kubernetes": 2015,
    "NumPy": 2006,
    "OpenJDK": 2007,
    "Pandas": 2009,
    "Python": 1994,
    "Rust": 2015,
    "Swift": 2014,
}


def get_governance_statistics(
    projects: List[IndividualProjectContext],
) -> List[GovernanceProjectStats]:
    """Calculates structured trend lines and multi-year profiles with strict typing."""

    # Mapping registry to link domains to their engine and target structural attributes
    computation_registry = {
        "independence": (compute_independence_hhi, "independence"),
        "independence_no_discard": (
            compute_independence_hhi_do_no_discard_and_no_bootstrapping,
            "independence_no_discard",
        ),
        "pluralism": (compute_pluralism_author_gini, "pluralism"),
        "representation": (compute_representation_comment_gini, "representation"),
        "decentralization": (compute_centralization_metrics, "decentralization"),
        "autonomous_participation": (
            compute_newcomers_onboarding,
            "autonomous_participation",
        ),
    }

    n_bootstraps = 200
    if os.getenv("DEBUG", "False").lower() == "true":
        print("DEBUG mode active: Reducing bootstrap iterations for faster execution.")
        n_bootstraps = 20

    project_records: List[GovernanceProjectStats] = []

    for ctx in projects:
        print(f"Analyzing structure variables for project: {ctx.project_name}")
        project_container = GovernanceProjectStats(
            project_name=ctx.project_name, release_year=RELEASE_YEARS[ctx.project_name]
        )

        for domain_key, (compute_fn, attribute_name) in computation_registry.items():
            # 1. Process and bind Time-Series Windows
            yearly_raw_dict = compute_fn(
                ctx, window_size=1, mode="all_windows", n_bootstraps=n_bootstraps
            )
            timeline_target = getattr(project_container.metrics, attribute_name)

            for year, interval_dict in yearly_raw_dict.items():
                timeline_target.windows[year] = MetricInterval(
                    val=interval_dict["val"],
                    ci_low=interval_dict["ci_low"],
                    ci_high=interval_dict["ci_high"],
                )

            # 2. Process and bind Summary Multi-Year Pooled metrics
            pooled_raw_dict = compute_fn(
                ctx, window_size=5, mode="most_recent", n_bootstraps=n_bootstraps
            )
            setattr(
                project_container.pooled_metrics,
                attribute_name,
                MetricInterval(
                    val=pooled_raw_dict["val"],
                    ci_low=pooled_raw_dict["ci_low"],
                    ci_high=pooled_raw_dict["ci_high"],
                ),
            )

        project_records.append(project_container)

    return project_records
