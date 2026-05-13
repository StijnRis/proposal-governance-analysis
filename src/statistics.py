"""Core statistics calculations for proposal governance analysis."""
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd


def get_projects(db_path: str) -> List[Tuple[int, str]]:
    """Get all projects from database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT project_id, project_name FROM Project")
    projects = cursor.fetchall()
    conn.close()
    return projects


def calculate_basic_counts(db_path: str, project_id: int) -> Dict:
    """Calculate basic counts: proposals, revisions, authors, comments, stages."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Count proposals
    cursor.execute("SELECT COUNT(*) FROM Proposal WHERE project_id = ?", (project_id,))
    num_proposals = cursor.fetchone()[0]

    # Count revisions
    cursor.execute("SELECT COUNT(*) FROM ProposalRevision WHERE project_id = ?", (project_id,))
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
        if status and any(keyword in str(status).lower() for keyword in success_keywords)
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

    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def get_author_tenure_distribution(db_path: str, project_id: int) -> Dict[str, List]:
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

    df["first_date"] = pd.to_datetime(df["first_date"])
    df["last_date"] = pd.to_datetime(df["last_date"])
    df["duration_days"] = (df["last_date"] - df["first_date"]).dt.days

    # Group by duration bins (in 30-day intervals)
    bins = list(range(0, int(df["duration_days"].max()) + 31, 30))
    tenure_dist = df["duration_days"].hist(bins=bins)

    return {
        "duration_days": df["duration_days"].tolist(),
        "author_count": [len(df[df["duration_days"].between(bins[i], bins[i + 1])]) for i in range(len(bins) - 1)],
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

    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def calculate_all_statistics(db_path: str) -> Dict:
    """Calculate all statistics for all projects."""
    projects = get_projects(db_path)
    all_stats = {}

    for project_id, project_name in projects:
        # per-proposal comment counts and unique authors
        comments_df = get_comments_per_proposal(db_path, project_id)
        if comments_df is None or comments_df.empty:
            avg_comments = 0.0
            avg_unique_authors = 0.0
        else:
            avg_comments = float(comments_df['comment_count'].mean())
            avg_unique_authors = float(comments_df['unique_authors'].mean())

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
        }
        all_stats[project_name] = stats

    return all_stats


def print_statistics_summary(stats: Dict) -> None:
    """Print a summary of calculated statistics."""
    for project_name, project_stats in stats.items():
        print(f"\n{'='*60}")
        print(f"Project: {project_name}")
        print(f"{'='*60}")

        counts = project_stats["basic_counts"]
        print(f"\nBasic Counts:")
        print(f"  Proposals: {counts['num_proposals']}")
        print(f"  Revisions: {counts['num_revisions']}")
        print(f"  Authors: {counts['num_authors']}")
        print(f"  Comments: {counts['num_comments']}")
        print(f"  Stages: {counts['num_stages']}")

        print(f"\nSuccess Rate: {project_stats['success_rate']:.2f}%")

