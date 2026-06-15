import datetime
from dataclasses import dataclass, field

import polars as pl
import pytest

# Import your code here. Assuming your code is in a module named 'governance_stats'
from governance_stats import (
    GovernanceProjectStats,
    MetricInterval,
    _bootstrap_interval,
    _polars_gini_expr,
    compute_independence_hhi,
    compute_pluralism_author_gini,
    get_governance_statistics,
)

# =====================================================================
# Dummy Context for Testing
# =====================================================================


@dataclass
class DummyProjectContext:
    """A lightweight mock mirroring IndividualProjectContext schema."""

    project_name: str
    proposal_revisions: pl.DataFrame = field(default_factory=pl.DataFrame)
    proposal_revision_authors: pl.DataFrame = field(default_factory=pl.DataFrame)
    comments: pl.DataFrame = field(default_factory=pl.DataFrame)
    affiliations: pl.DataFrame = field(default_factory=pl.DataFrame)
    organisations: pl.DataFrame = field(default_factory=pl.DataFrame)


@pytest.fixture
def empty_context():
    """Returns a completely empty context with proper schemas to satisfy the query planner."""
    return DummyProjectContext(
        project_name="Empty Project",
        proposal_revisions=pl.DataFrame(
            {"proposal_id": [], "revision_index": [], "created_at": []},
            schema={
                "proposal_id": pl.Int64,
                "revision_index": pl.Int64,
                "created_at": pl.Datetime,  # <--- This prevents the pl.Null crash!
            },
        ),
        proposal_revision_authors=pl.DataFrame(
            {"proposal_id": [], "revision_index": [], "author_id": []},
            schema={
                "proposal_id": pl.Int64,
                "revision_index": pl.Int64,
                "author_id": pl.Int64,
            },
        ),
        comments=pl.DataFrame(
            {"proposal_id": [], "author_id": [], "created_at": []},
            schema={
                "proposal_id": pl.Int64,
                "author_id": pl.Int64,
                "created_at": pl.Datetime,  # <--- Fix here too for representation/centralization
            },
        ),
        affiliations=pl.DataFrame(
            {"person_id": [], "organisation_id": []},
            schema={"person_id": pl.Int64, "organisation_id": pl.Int64},
        ),
        organisations=pl.DataFrame(
            {"organisation_id": [], "organisation_name": []},
            schema={"organisation_id": pl.Int64, "organisation_name": pl.String},
        ),
    )


@pytest.fixture
def populated_context():
    """Returns a standard populated context spanning 2024 to 2025."""
    # 2 Proposals
    revisions = pl.DataFrame(
        {
            "proposal_id": [1, 2],
            "revision_index": [0, 0],
            "created_at": [
                datetime.datetime(2024, 5, 1),
                datetime.datetime(2025, 6, 1),
            ],
        }
    )

    # Authors (Author 10 and 20 work on Prop 1; Author 20 works on Prop 2)
    authors = pl.DataFrame(
        {
            "proposal_id": [1, 1, 2],
            "revision_index": [0, 0, 0],
            "author_id": [10, 20, 20],
        }
    )

    # Comments
    comments = pl.DataFrame(
        {
            "proposal_id": [1, 2],
            "author_id": [10, 30],  # Author 30 only comments
            "created_at": [
                datetime.datetime(2024, 5, 2),
                datetime.datetime(2025, 6, 2),
            ],
        }
    )

    # Affiliations & Orgs
    affiliations = pl.DataFrame(
        {"person_id": [10, 20, 30], "organisation_id": [100, 200, 100]}
    )

    orgs = pl.DataFrame(
        {"organisation_id": [100, 200], "organisation_name": ["Org A", "Org B"]}
    )

    return DummyProjectContext(
        project_name="Alpha Project",
        proposal_revisions=revisions,
        proposal_revision_authors=authors,
        comments=comments,
        affiliations=affiliations,
        organisations=orgs,
    )


# =====================================================================
# Unit Tests: Atomic Mathematical Helpers
# =====================================================================


def test_polars_gini_expr_perfect_equality():
    """An equally distributed list should yield a Gini coefficient of 0.0."""
    df = pl.DataFrame({"shares": [10, 10, 10, 10]})
    res = df.select(_polars_gini_expr("shares"))
    assert res.item() == 0.0


def test_polars_gini_expr_empty_or_single():
    """Dataframes with 0 or 1 rows should default to 0.0 cleanly."""
    df_empty = pl.DataFrame({"shares": pl.Series([], dtype=pl.Int64)})
    df_single = pl.DataFrame({"shares": [100]})

    assert df_empty.select(_polars_gini_expr("shares")).item() == 0.0
    assert df_single.select(_polars_gini_expr("shares")).item() == 0.0


def test_bootstrap_interval_fallback():
    """If unique clusters < 5, bootstrap should skip execution and return the point estimate."""
    df = pl.DataFrame({"cluster": [1, 2, 3], "value": [10, 20, 30]})

    def dummy_calc(sliced_df):
        return sliced_df["value"].mean()

    res = _bootstrap_interval(df, dummy_calc, cluster_col="cluster", n_bootstraps=10)

    # Point estimate mean calculation: (10+20+30)/3 = 20.0
    assert res["val"] == 20.0
    assert res["ci_low"] == 20.0
    assert res["ci_high"] == 20.0


# =====================================================================
# Functional Tests: Metric Engines
# =====================================================================


def test_compute_independence_hhi_empty(empty_context):
    """Verifies HHI engine handles empty contexts gracefully without crashing."""
    res_recent = compute_independence_hhi(
        empty_context, window_size=1, mode="most_recent"
    )
    res_all = compute_independence_hhi(empty_context, window_size=1, mode="all_windows")

    assert res_recent == {"val": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    assert res_all == {}


def test_compute_pluralism_invalid_mode(populated_context):
    """Passing an unsupported execution mode should trigger a ValueError."""
    with pytest.raises(ValueError, match="Unknown mode: unexpected_mode"):
        compute_pluralism_author_gini(
            populated_context, window_size=1, mode="unexpected_mode"
        )


def test_compute_pluralism_all_windows(populated_context):
    """Validates structure and baseline values returned for multi-window calculations."""
    # Use low n_bootstraps to make tests run fast
    res = compute_pluralism_author_gini(
        populated_context, window_size=1, mode="all_windows", n_bootstraps=5
    )

    # Our populated context has events in 2024 and 2025
    assert 2024 in res
    assert 2025 in res
    assert "val" in res[2024]
    assert "ci_low" in res[2024]


# =====================================================================
# Integration Tests: Pipeline Orchestration
# =====================================================================


def test_get_governance_statistics_pipeline(populated_context):
    """Tests the full orchestration step looping over a list of projects."""
    # Overwrite network intensive bootstrap scaling down for unit testing speed
    results = get_governance_statistics([populated_context])

    assert len(results) == 1
    project_record = results[0]

    assert isinstance(project_record, GovernanceProjectStats)
    assert project_record.project_name == "Alpha Project"

    # Check that pooled metrics bound successfully (with default window_size=5)
    assert isinstance(project_record.pooled_metrics.independence, MetricInterval)
    assert isinstance(project_record.pooled_metrics.pluralism, MetricInterval)

    # Verify time-series timeline maps built successfully
    assert len(project_record.metrics.independence.windows) > 0
