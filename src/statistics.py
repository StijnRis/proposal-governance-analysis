"""Core statistics calculations for proposal governance analysis."""

import sqlite3
from typing import Any, Dict, List, Tuple

import pandas as pd

# use shared parser
from timeutils import to_naive_series as _to_naive_series


def get_projects(db_path: str) -> List[Tuple[int, str]]:
    """Get all projects from database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT project_id, project_name FROM Project")
    projects = cursor.fetchall()
    conn.close()
    return projects


def calculate_basic_counts(db_path: str, project_id: int) -> Dict[str, int]:
    """Calculate basic counts: proposals, revisions, authors, comments, stages."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Count proposals
    cursor.execute("SELECT COUNT(*) FROM Proposal WHERE project_id = ?", (project_id,))
    num_proposals = cursor.fetchone()[0]

    # Count revisions
    cursor.execute(
        "SELECT COUNT(*) FROM ProposalRevision WHERE project_id = ?", (project_id,)
    )
    num_revisions = cursor.fetchone()[0]

    # Count unique authors across all revisions
    cursor.execute(
        "SELECT COUNT(DISTINCT author_id) FROM ProposalRevisionAuthor WHERE project_id = ?",
        (project_id,),
    )
    num_authors = cursor.fetchone()[0]

    # Count comments
    cursor.execute("SELECT COUNT(*) FROM Comment WHERE project_id = ?", (project_id,))
    num_comments = cursor.fetchone()[0]

    # Count unique stages
    cursor.execute(
        "SELECT COUNT(DISTINCT normalised_status) FROM StageHistory WHERE project_id = ?",
        (project_id,),
    )
    num_stages = cursor.fetchone()[0]

    conn.close()

    return {
        "num_proposals": num_proposals,
        "num_revisions": num_revisions,
        "num_authors": num_authors,
        "num_comments": num_comments,
        "num_stages": num_stages,
    }


def calculate_success_rate(db_path: str, project_id: int) -> float:
    """Calculate success rate of proposals based on final stage."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get proposals with their final status
    cursor.execute(
        """
        SELECT p.proposal_id, 
               (SELECT normalised_status FROM StageHistory 
                WHERE proposal_id = p.proposal_id AND project_id = ?
                ORDER BY created_at DESC LIMIT 1) as final_status
        FROM Proposal p
        WHERE p.project_id = ?
        """,
        (project_id, project_id),
    )
    proposals = cursor.fetchall()
    conn.close()

    if not proposals:
        return 0.0

    # Count successful proposals (assuming "approved" or similar indicates success)
    success_keywords = ["approved", "accepted", "merged"]
    successful = sum(
        1
        for _, status in proposals
        if status
        and any(keyword in str(status).lower() for keyword in success_keywords)
    )

    return (successful / len(proposals)) * 100 if proposals else 0.0


def get_revisions_over_time(db_path: str, project_id: int) -> pd.DataFrame:
    """Get proposal revisions over time."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT proposal_id, created_at, revision_index
        FROM ProposalRevision
        WHERE project_id = ?
        ORDER BY created_at
    """
    df = pd.read_sql_query(query, conn, params=(project_id,))
    conn.close()

    if df.empty:
        return df

    df["created_at"] = _to_naive_series(df["created_at"])
    return df


def get_author_tenure_distribution(
    db_path: str, project_id: int
) -> Dict[str, List[int]]:
    """Calculate how long each author stayed active (first to last revision)."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT author_id, MIN(created_at) as first_date, MAX(created_at) as last_date
        FROM (
            SELECT author_id, created_at
            FROM ProposalRevisionAuthor pra
            JOIN ProposalRevision pr ON pra.proposal_id = pr.proposal_id 
                AND pra.revision_index = pr.revision_index
                AND pra.project_id = pr.project_id
            WHERE pra.project_id = ?
        )
        GROUP BY author_id
    """
    df = pd.read_sql_query(query, conn, params=(project_id,))
    conn.close()

    if df.empty:
        return {"duration_days": [], "author_count": []}

    df["first_date"] = _to_naive_series(df["first_date"])
    df["last_date"] = _to_naive_series(df["last_date"])
    df["duration_days"] = (df["last_date"] - df["first_date"]).dt.days

    # Group by duration bins (in 30-day intervals)
    bins = list(range(0, int(df["duration_days"].max()) + 31, 30))
    tenure_dist = df["duration_days"].hist(bins=bins)

    return {
        "duration_days": df["duration_days"].tolist(),
        "author_count": [
            len(df[df["duration_days"].between(bins[i], bins[i + 1])])
            for i in range(len(bins) - 1)
        ],
        "bins": bins,
    }


def get_author_activity_distribution(db_path: str, project_id: int) -> pd.DataFrame:
    """Get author activity sorted by most active first."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT author_id, COUNT(*) as revision_count
        FROM ProposalRevisionAuthor
        WHERE project_id = ?
        GROUP BY author_id
        ORDER BY revision_count DESC
    """
    df = pd.read_sql_query(query, conn, params=(project_id,))
    conn.close()

    return df


def get_comments_per_proposal(db_path: str, project_id: int) -> pd.DataFrame:
    """Get comment counts and unique authors per proposal."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT proposal_id, 
               COUNT(*) as comment_count,
               COUNT(DISTINCT author_id) as unique_authors
        FROM Comment
        WHERE project_id = ? AND proposal_id IS NOT NULL
        GROUP BY proposal_id
        ORDER BY comment_count DESC
    """
    df = pd.read_sql_query(query, conn, params=(project_id,))
    conn.close()

    return df


def get_revision_authors_per_year(db_path: str, project_id: int) -> Dict[int, int]:
    """Return a mapping year -> number of unique authors who proposed revisions that year."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT pra.author_id, pr.created_at
        FROM ProposalRevisionAuthor pra
        JOIN ProposalRevision pr ON pra.proposal_id = pr.proposal_id
            AND pra.revision_index = pr.revision_index
            AND pra.project_id = pr.project_id
        WHERE pra.project_id = ?
    """
    df = pd.read_sql_query(query, conn, params=(project_id,))
    conn.close()

    if df.empty:
        return {}

    df["created_at"] = _to_naive_series(df["created_at"])
    df["year"] = df["created_at"].dt.year
    counts = df.groupby("year")["author_id"].nunique().sort_index()
    return {int(k): int(v) for k, v in counts.to_dict().items()}


def get_comment_authors_per_year(db_path: str, project_id: int) -> Dict[int, int]:
    """Return a mapping year -> number of unique authors who left comments that year."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT author_id, created_at
        FROM Comment
        WHERE project_id = ?
    """
    df = pd.read_sql_query(query, conn, params=(project_id,))
    conn.close()

    if df.empty:
        return {}

    df["created_at"] = _to_naive_series(df["created_at"])
    df["year"] = df["created_at"].dt.year
    counts = df.groupby("year")["author_id"].nunique().sort_index()
    return {int(k): int(v) for k, v in counts.to_dict().items()}


def get_proposal_stage_counts_per_year(
    db_path: str, project_id: int
) -> Dict[int, Dict[str, int]]:
    """Return mapping year -> {stage: number_of_proposals_in_stage}.

    For each proposal and each year, the stage is taken as the latest `normalised_status`
    recorded on or before the end of that year. Proposals without any stage before
    a given year are ignored for that year.
    """
    conn = sqlite3.connect(db_path)
    query = """
        SELECT proposal_id, normalised_status, created_at
        FROM StageHistory
        WHERE project_id = ?
        ORDER BY proposal_id, created_at
    """
    df = pd.read_sql_query(query, conn, params=(project_id,))
    conn.close()

    if df.empty:
        return {}

    df["created_at"] = _to_naive_series(df["created_at"])
    df = df.dropna(subset=["created_at"]).copy()
    if df.empty:
        return {}

    df["year"] = df["created_at"].dt.year

    # collect all years of interest
    years = sorted(df["year"].unique().tolist())

    # build mapping proposal_id -> list of (created_at, stage)
    proposals = {}
    for _, row in df.iterrows():
        pid = row["proposal_id"]
        proposals.setdefault(pid, []).append(
            (row["created_at"], str(row["normalised_status"]))
        )

    # for each year, determine each proposal's last stage on or before year-end
    result: Dict[int, Dict[str, int]] = {}
    for year in years:
        year_end = pd.to_datetime(f"{year}-12-31 23:59:59")
        stage_counts: Dict[str, int] = {}
        for pid, events in proposals.items():
            # events are in chronological order due to SQL ORDER BY
            last_stage = None
            for ts, stage in events:
                if pd.isna(ts):
                    continue
                if ts <= year_end:
                    last_stage = stage
                else:
                    break
            if last_stage is not None:
                stage_counts[last_stage] = stage_counts.get(last_stage, 0) + 1

        result[year] = stage_counts

    return result


def get_proposals_over_time(db_path: str, project_id: int) -> pd.DataFrame:
    """Get proposal creation dates (earliest revision) for proposals in a project."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT proposal_id, MIN(created_at) as created_at
        FROM ProposalRevision
        WHERE project_id = ?
        GROUP BY proposal_id
        ORDER BY created_at
    """
    df = pd.read_sql_query(query, conn, params=(project_id,))
    conn.close()

    if df.empty:
        return df

    df["created_at"] = _to_naive_series(df["created_at"])
    return df


def calculate_all_statistics(db_path: str) -> Dict[str, Dict[str, Any]]:
    """Calculate all statistics and governance metrics for all projects."""
    projects = get_projects(db_path)
    all_stats: Dict[str, Dict[str, Any]] = {}

    # import governance helpers lazily to avoid import-time issues
    import governance

    for project_id, project_name in projects:
        # per-proposal comment counts and unique authors
        comments_df = get_comments_per_proposal(db_path, project_id)
        if comments_df is None or comments_df.empty:
            avg_comments = 0.0
            avg_unique_authors = 0.0
        else:
            avg_comments = float(comments_df["comment_count"].mean())
            avg_unique_authors = float(comments_df["unique_authors"].mean())

        # core stats
        stats = {
            "project_id": project_id,
            "project_name": project_name,
            "basic_counts": calculate_basic_counts(db_path, project_id),
            "success_rate": calculate_success_rate(db_path, project_id),
            "revisions_over_time": get_revisions_over_time(db_path, project_id),
            "proposals_over_time": get_proposals_over_time(db_path, project_id),
            "comments_per_proposal": comments_df,
            "avg_comments_per_proposal": avg_comments,
            "avg_unique_authors_per_proposal": avg_unique_authors,
            # authors per year metrics
            "authors_proposing_per_year": get_revision_authors_per_year(
                db_path, project_id
            ),
            "authors_commenting_per_year": get_comment_authors_per_year(
                db_path, project_id
            ),
            "proposal_stage_counts_per_year": get_proposal_stage_counts_per_year(
                db_path, project_id
            ),
        }

        # governance metrics per year
        independence = governance.compute_independence_hhi_per_year(db_path, project_id)
        pluralism = governance.compute_pluralism_author_gini_per_year(
            db_path, project_id
        )
        representation = governance.compute_representation_comment_gini_per_year(
            db_path, project_id
        )
        centralization = governance.compute_betweenness_centralization_per_year(
            db_path, project_id
        )
        newcomer_success = governance.compute_newcomer_success_rate_per_year(
            db_path, project_id
        )

        years = sorted(
            set().union(
                independence.keys(),
                pluralism.keys(),
                representation.keys(),
                centralization.keys(),
                newcomer_success.keys(),
            )
        )

        governance_metrics = {
            "years": years,
            "independence_hhi": independence,
            "pluralism_gini": pluralism,
            "representation_gini": representation,
            "centralization": centralization,
            # newcomer success returned as fraction between 0 and 1
            "newcomer_success": newcomer_success,
        }

        stats["governance_metrics"] = governance_metrics
        all_stats[project_name] = stats

    return all_stats


def print_statistics_summary(stats: Dict) -> None:
    """Print a summary of calculated statistics."""
    for project_name, project_stats in stats.items():
        print(f"\n{'=' * 60}")
        print(f"Project: {project_name}")
        print(f"{'=' * 60}")

        counts = project_stats["basic_counts"]
        print(f"\nBasic Counts:")
        print(f"  Proposals: {counts['num_proposals']}")
        print(f"  Revisions: {counts['num_revisions']}")
        print(f"  Authors: {counts['num_authors']}")
        print(f"  Comments: {counts['num_comments']}")
        print(f"  Stages: {counts['num_stages']}")

        print(f"\nSuccess Rate: {project_stats['success_rate']:.2f}%")
