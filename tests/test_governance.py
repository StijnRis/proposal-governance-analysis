import datetime
from dataclasses import dataclass, field
from unittest.mock import patch

import polars as pl
import pytest

# Import your code here. Assuming your code is in a module named 'governance_stats'
from governance_calc import (
    RELEASE_YEARS,
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
                "created_at": pl.Datetime, 
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
                "created_at": pl.Datetime,
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
    """Returns a standard populated context spanning 2024 to 2026 to safely manage max_year - 1 boundaries."""
    # 3 Proposals
    revisions = pl.DataFrame(
        {
            "proposal_id": [1, 2, 3],
            "revision_index": [0, 0, 0],
            "created_at": [
                datetime.datetime(2024, 5, 1),
                datetime.datetime(2025, 6, 1),
                datetime.datetime(2026, 7, 1),
            ],
        }
    )

    # Authors (Author 10 and 20 work on Prop 1; Author 20 works on Prop 2; Author 10 on Prop 3)
    authors = pl.DataFrame(
        {
            "proposal_id": [1, 1, 2, 3],
            "revision_index": [0, 0, 0, 0],
            "author_id": [10, 20, 20, 10],
        }
    )

    # Comments
    comments = pl.DataFrame(
        {
            "proposal_id": [1, 2, 3],
            "author_id": [10, 30, 10],  # Author 30 only comments
            "created_at": [
                datetime.datetime(2024, 5, 2),
                datetime.datetime(2025, 6, 2),
                datetime.datetime(2026, 7, 2),
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


def test_bootstrap_interval_execution_and_bounds():
    """Verifies that bootstrap runs resampling when clusters >= 5 and bounds are sane."""
    # Data with 6 unique clusters (1 through 6) to pass the cluster len check
    df = pl.DataFrame({
        "cluster": [1, 2, 3, 4, 5, 6],
        "value": [10.0, 15.0, 50.0, 55.0, 90.0, 100.0]
    })

    def calc_mean(sliced_df):
        return sliced_df["value"].mean()

    # Run with a 95% CI
    res = _bootstrap_interval(
        df, 
        calc_mean, 
        cluster_col="cluster", 
        n_bootstraps=100, 
        ci=0.95
    )

    # 1. Assert the point estimate is calculated correctly
    assert res["val"] == pytest.approx(53.3333, rel=1e-4)
    
    # 2. Assert variance bounds remain strictly inside the logical range of data
    assert res["ci_low"] >= 10.0
    assert res["ci_high"] <= 100.0


def test_bootstrap_interval_deterministic_seeding():
    """Asserts that the internal seed management renders perfectly reproducible results."""
    df = pl.DataFrame({
        "cluster": [1, 2, 3, 4, 5, 6],
        "value": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    })
    
    def calc_mean(sliced_df):
        return sliced_df["value"].mean()

    # Execute twice independently
    res_first = _bootstrap_interval(df, calc_mean, cluster_col="cluster", n_bootstraps=50)
    res_second = _bootstrap_interval(df, calc_mean, cluster_col="cluster", n_bootstraps=50)

    # Because you used np.random.default_rng(42) internally, 
    # these values must match down to the decimal point across test invocations.
    assert res_first["ci_low"] == res_second["ci_low"]
    assert res_first["ci_high"] == res_second["ci_high"]


# =====================================================================
# Functional Tests: Metric Engines
# =====================================================================


def test_compute_independence_hhi_empty(empty_context):
    """Verifies HHI engine handles empty contexts gracefully without crashing."""
    res_recent = compute_independence_hhi(
        empty_context, window_size=1, mode="most_recent", n_bootstraps=5
    )
    res_all = compute_independence_hhi(
        empty_context, window_size=1, mode="all_windows", n_bootstraps=5
    )

    assert res_recent == {"val": 0.0, "ci_low": None, "ci_high": None}
    assert res_all == {}


def test_compute_pluralism_invalid_mode(populated_context):
    """Passing an unsupported execution mode should trigger a ValueError."""
    with pytest.raises(ValueError, match="Unknown mode: unexpected_mode"):
        compute_pluralism_author_gini(
            populated_context, window_size=1, mode="unexpected_mode", n_bootstraps=5
        )


def test_compute_pluralism_all_windows(populated_context):
    """Validates structure and baseline values returned for multi-window calculations."""
    res = compute_pluralism_author_gini(
        populated_context, window_size=1, mode="all_windows", n_bootstraps=5
    )

    # Our populated context has events spanning up to 2026. 
    # compute_pluralism_author_gini does not subtract 1 from max_year.
    assert 2024 in res
    assert 2025 in res
    assert 2026 in res
    assert "val" in res[2024]
    assert "ci_low" in res[2024]


# =====================================================================
# Integration Tests: Pipeline Orchestration
# =====================================================================


def test_get_governance_statistics_pipeline(populated_context):
    """Tests the full orchestration step looping over a list of projects."""
    
    # Mock RELEASE_YEARS to include our test project name
    mocked_release_years = {**RELEASE_YEARS, "Alpha Project": 2024}
    
    # Patch the RELEASE_YEARS dictionary in the governance_calc module
    with patch("governance_calc.RELEASE_YEARS", mocked_release_years):
        results = get_governance_statistics([populated_context])

    assert len(results) == 1
    project_record = results[0]

    assert project_record.project_name == "Alpha Project"
    assert project_record.release_year == 2024

    # Check that pooled summary structures bound successfully
    assert isinstance(project_record.pooled_metrics.independence, MetricInterval)
    assert isinstance(project_record.pooled_metrics.pluralism, MetricInterval)