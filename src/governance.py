"""Governance metric computations per year for projects.

Provides functions called by statistics.calculate_all_statistics to compute
per-year values for the five governance dimensions described in the proposal.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Union

import networkx as nx
import pandas as pd


def _parse_datetime_series(series):
    """Robustly parse a pandas Series of datetimes (strings, bytes) into UTC datetimes."""

    return series.apply(_parse_single_datetime)


def _parse_single_datetime(value):
    """Parse a single datetime value (string or bytes) into a pandas.Timestamp in UTC."""
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:
            return pd.NaT
    # numeric epoch handling
    if isinstance(value, (int, float)) or (
        isinstance(value, str)
        and value.strip().lstrip("-").replace(".", "", 1).isdigit()
    ):
        val = float(value)
        if val > 1e15:
            return pd.to_datetime(int(val), unit="ns", utc=True, errors="coerce")
        if val > 1e12:
            return pd.to_datetime(int(val), unit="ms", utc=True, errors="coerce")
        return pd.to_datetime(val, unit="s", utc=True, errors="coerce")

    return pd.to_datetime(value, utc=True, errors="coerce")


def _gini(values: List[float]) -> float:
    if not values:
        return 0.0
    arr = pd.Series(values, dtype=float).values
    if arr.sum() == 0:
        return 0.0
    n = len(arr)
    mean = arr.mean()
    diff_sum = 0.0
    for i in range(n):
        diff_sum += abs(arr[i] - arr).sum()
    return float(diff_sum / (2 * (n**2) * mean))


DBOrConn = Union[Path, sqlite3.Connection]


def _open_conn(db: DBOrConn):
    if isinstance(db, sqlite3.Connection):
        return db, False
    return sqlite3.connect(str(db)), True


def _get_person_org_map(
    conn: sqlite3.Connection, person_ids: List[int]
) -> Dict[int, str]:
    if not person_ids:
        return {}
    cursor = conn.cursor()
    result = {pid: None for pid in person_ids}

    # Try Affiliation -> Organisation
    q = f"SELECT a.person_id, o.organisation_name FROM Affiliation a JOIN Organisation o ON a.organisation_id = o.organisation_id WHERE a.person_id IN ({','.join(['?'] * len(person_ids))})"
    cursor.execute(q, person_ids)
    for person_id, org_name in cursor.fetchall():
        if org_name:
            result[person_id] = org_name

    remaining = [pid for pid, v in result.items() if v is None]
    if remaining:
        # fall back to PersonIdentifier table (new schema)
        q2 = f"SELECT person_id, domain FROM PersonIdentifier WHERE person_id IN ({','.join(['?'] * len(remaining))})"
        cursor.execute(q2, remaining)
        # there may be multiple identifiers per person; prefer any domain we find
        for person_id, domain in cursor.fetchall():
            if result.get(person_id) is None and domain:
                result[person_id] = domain

    for pid in result:
        if result[pid] is None:
            result[pid] = f"unknown_{pid}"

    return result


def compute_independence_hhi_per_year(
    db_path: DBOrConn, project_id: int
) -> Dict[int, float]:
    conn, close = _open_conn(db_path)
    proposals_df = pd.read_sql_query(
        "SELECT proposal_id, MIN(created_at) as created_at FROM ProposalRevision WHERE project_id = ? GROUP BY proposal_id",
        conn,
        params=(project_id,),
    )
    if proposals_df.empty:
        conn.close()
        return {}
    proposals_df["created_at"] = _parse_datetime_series(proposals_df["created_at"])
    proposals_df["year"] = proposals_df["created_at"].dt.year
    years = sorted(proposals_df["year"].unique())
    year_hhi: Dict[int, float] = {}
    cursor = conn.cursor()

    for year in years:
        props = proposals_df[proposals_df["year"] == year]["proposal_id"].tolist()
        if not props:
            year_hhi[year] = 0.0
            continue
        q = f"SELECT DISTINCT author_id FROM ProposalRevisionAuthor WHERE project_id = ? AND proposal_id IN ({','.join(['?'] * len(props))})"
        params = [project_id] + props
        cursor.execute(q, params)
        rows = cursor.fetchall()
        person_ids = [r[0] for r in rows]
        if not person_ids:
            year_hhi[year] = 0.0
            continue
        # Map persons to organisations and exclude unmapped/unknown authors from HHI
        person_org = _get_person_org_map(conn, person_ids)
        org_counts: Dict[str, int] = {}
        for pid in person_ids:
            org = person_org.get(pid)
            # drop unmapped/unknown authors (these are not organizational affiliations)
            if org is None:
                continue
            if isinstance(org, str) and org.startswith("unknown_"):
                continue
            org_counts[org] = org_counts.get(org, 0) + 1
        total = sum(org_counts.values())
        if total == 0:
            year_hhi[year] = 0.0
            continue
        shares = [c / total for c in org_counts.values()]
        year_hhi[year] = float(sum(s * s for s in shares))

    if close:
        conn.close()
    return year_hhi


def compute_pluralism_author_gini_per_year(
    db_path: DBOrConn, project_id: int
) -> Dict[int, float]:
    conn, close = _open_conn(db_path)
    proposals_df = pd.read_sql_query(
        "SELECT proposal_id, MIN(created_at) as created_at FROM ProposalRevision WHERE project_id = ? GROUP BY proposal_id",
        conn,
        params=(project_id,),
    )
    if proposals_df.empty:
        conn.close()
        return {}
    proposals_df["created_at"] = _parse_datetime_series(proposals_df["created_at"])
    proposals_df["year"] = proposals_df["created_at"].dt.year
    years = sorted(proposals_df["year"].unique())
    cursor = conn.cursor()
    year_gini: Dict[int, float] = {}

    for year in years:
        props = proposals_df[proposals_df["year"] == year]["proposal_id"].tolist()
        if not props:
            year_gini[year] = 0.0
            continue
        authors: List[int] = []
        for pid in props:
            cursor.execute(
                "SELECT MIN(revision_index) FROM ProposalRevision WHERE project_id = ? AND proposal_id = ?",
                (project_id, pid),
            )
            min_rev = cursor.fetchone()[0]
            if min_rev is None:
                continue
            cursor.execute(
                "SELECT DISTINCT author_id FROM ProposalRevisionAuthor WHERE project_id = ? AND proposal_id = ? AND revision_index = ?",
                (project_id, pid, min_rev),
            )
            authors.extend([r[0] for r in cursor.fetchall()])
        if not authors:
            year_gini[year] = 0.0
            continue
        counts: Dict[int, int] = {}
        for a in authors:
            counts[a] = counts.get(a, 0) + 1
        year_gini[year] = _gini(list(counts.values()))

    if close:
        conn.close()
    return year_gini


def compute_representation_comment_gini_per_year(
    db_path: DBOrConn, project_id: int
) -> Dict[int, float]:
    conn, close = _open_conn(db_path)
    df = pd.read_sql_query(
        "SELECT author_id, created_at FROM Comment WHERE project_id = ? AND author_id IS NOT NULL",
        conn,
        params=(project_id,),
    )
    if df.empty:
        conn.close()
        return {}
    df["created_at"] = _parse_datetime_series(df["created_at"])
    df["year"] = df["created_at"].dt.year
    years = sorted(df["year"].unique())
    year_gini: Dict[int, float] = {}
    for year in years:
        sub = df[df["year"] == year]
        if sub.empty:
            year_gini[year] = 0.0
            continue
        counts = sub.groupby("author_id").size().tolist()
        year_gini[year] = _gini(counts)
    if close:
        conn.close()
    return year_gini


def compute_betweenness_centralization_per_year(
    db_path: DBOrConn, project_id: int
) -> Dict[int, float]:
    conn, close = _open_conn(db_path)
    rev_df = pd.read_sql_query(
        "SELECT proposal_id, revision_index, created_at FROM ProposalRevision WHERE project_id = ?",
        conn,
        params=(project_id,),
    )
    if rev_df.empty:
        conn.close()
        return {}
    rev_df["created_at"] = _parse_datetime_series(rev_df["created_at"])
    rev_df["year"] = rev_df["created_at"].dt.year

    comment_df = pd.read_sql_query(
        "SELECT proposal_id, author_id, created_at FROM Comment WHERE project_id = ? AND proposal_id IS NOT NULL AND author_id IS NOT NULL",
        conn,
        params=(project_id,),
    )
    if not comment_df.empty:
        comment_df["created_at"] = _parse_datetime_series(comment_df["created_at"])
        comment_df["year"] = comment_df["created_at"].dt.year
    else:
        comment_df = pd.DataFrame(
            columns=["proposal_id", "author_id", "created_at", "year"]
        )

    years = sorted(pd.concat([rev_df["year"], comment_df["year"]]).dropna().unique())
    year_centralization: Dict[int, float] = {}
    cursor = conn.cursor()

    for year in years:
        weights = {}
        nodes = set()
        revs_in_year = rev_df[rev_df["year"] == year]
        for _, row in revs_in_year.iterrows():
            proposal_id = row["proposal_id"]
            rev_idx = int(row["revision_index"])
            cursor.execute(
                "SELECT DISTINCT author_id FROM ProposalRevisionAuthor WHERE project_id = ? AND proposal_id = ? AND revision_index = ?",
                (project_id, proposal_id, rev_idx),
            )
            authors = [r[0] for r in cursor.fetchall()]
            for a in authors:
                nodes.add(a)
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    a, b = authors[i], authors[j]
                    key = (min(a, b), max(a, b))
                    weights[key] = weights.get(key, 0) + 1
        comments_in_year = comment_df[comment_df["year"] == year]
        if not comments_in_year.empty:
            grouped = comments_in_year.groupby("proposal_id")
            for pid, group in grouped:
                authors = group["author_id"].astype(int).unique().tolist()
                for a in authors:
                    nodes.add(a)
                for i in range(len(authors)):
                    for j in range(i + 1, len(authors)):
                        a, b = authors[i], authors[j]
                        key = (min(a, b), max(a, b))
                        weights[key] = weights.get(key, 0) + 1
        if not nodes:
            year_centralization[year] = 0.0
            continue
        G = nx.Graph()
        G.add_nodes_from(list(nodes))
        for (a, b), w in weights.items():
            # convert interaction strength to path distance (higher interaction => shorter distance)
            dist = (1.0 / w) if w > 0 else float("inf")
            G.add_edge(a, b, distance=dist)
        if G.number_of_nodes() < 3:
            year_centralization[year] = 0.0
            continue
        # compute normalized betweenness centrality using the distance attribute
        centrality = nx.betweenness_centrality(G, weight="distance", normalized=True)
        if not centrality:
            year_centralization[year] = 0.0
            continue
        max_cb = max(centrality.values())
        n = G.number_of_nodes()
        sum_diff = sum((max_cb - v) for v in centrality.values())
        # normalize by (n-1)(n-2) to match standard graph-level betweenness centralization
        # (see Freeman 1979). For small graphs this will be 0.0.
        denom = (n - 1) * (n - 2)
        centralization = sum_diff / denom if denom > 0 else 0.0
        year_centralization[year] = float(centralization)

    if close:
        conn.close()
    return year_centralization


def compute_newcomer_success_rate_per_year(
    db_path: DBOrConn, project_id: int
) -> Dict[int, float]:
    conn, close = _open_conn(db_path)
    proposals_df = pd.read_sql_query(
        "SELECT proposal_id, MIN(created_at) as created_at FROM ProposalRevision WHERE project_id = ? GROUP BY proposal_id",
        conn,
        params=(project_id,),
    )
    if proposals_df.empty:
        conn.close()
        return {}
    proposals_df["created_at"] = _parse_datetime_series(proposals_df["created_at"])
    proposals_df["year"] = proposals_df["created_at"].dt.year
    years = sorted(proposals_df["year"].unique())
    cursor = conn.cursor()
    year_success: Dict[int, float] = {}

    for year in years:
        subs = proposals_df[proposals_df["year"] == year]
        # FIX: compute onboarding rate = share of proposals in the year initiated by first-time authors
        total_proposals_in_year = len(subs)
        newcomer_proposals_count = 0
        for _, prow in subs.iterrows():
            pid = prow["proposal_id"]
            created_at = prow["created_at"]
            cursor.execute(
                "SELECT MIN(revision_index) FROM ProposalRevision WHERE project_id = ? AND proposal_id = ?",
                (project_id, pid),
            )
            min_rev = cursor.fetchone()[0]
            if min_rev is None:
                continue
            cursor.execute(
                "SELECT DISTINCT author_id FROM ProposalRevisionAuthor WHERE project_id = ? AND proposal_id = ? AND revision_index = ?",
                (project_id, pid, min_rev),
            )
            first_authors = [r[0] for r in cursor.fetchall()]
            if not first_authors:
                continue
            is_newcomer = True
            for a in first_authors:
                cursor.execute(
                    "SELECT MIN(pr.created_at) FROM ProposalRevision pr JOIN ProposalRevisionAuthor pra ON pr.project_id = pra.project_id AND pr.proposal_id = pra.proposal_id AND pr.revision_index = pra.revision_index WHERE pra.author_id = ?",
                    (a,),
                )
                first_activity = cursor.fetchone()[0]
                if first_activity is not None:
                    if _parse_single_datetime(first_activity) < created_at:
                        is_newcomer = False
                        break
            if is_newcomer:
                newcomer_proposals_count += 1
        if total_proposals_in_year == 0:
            year_success[year] = 0.0
        else:
            year_success[year] = newcomer_proposals_count / total_proposals_in_year

    if close:
        conn.close()
    return year_success
