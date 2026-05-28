import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from src import governance


def create_db(path: Path):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE ProposalRevision (project_id INTEGER, proposal_id INTEGER, revision_index INTEGER, created_at TEXT);
        CREATE TABLE ProposalRevisionAuthor (project_id INTEGER, proposal_id INTEGER, revision_index INTEGER, author_id INTEGER);
        CREATE TABLE Comment (project_id INTEGER, proposal_id INTEGER, author_id INTEGER, created_at TEXT);
        CREATE TABLE Affiliation (organisation_id INTEGER, person_id INTEGER);
        CREATE TABLE Organisation (organisation_id INTEGER, organisation_name TEXT);
        CREATE TABLE PersonIdentifier (person_id INTEGER, domain TEXT, identifier_type TEXT, identifier TEXT);
        """
    )
    conn.commit()
    return conn


def test_independence_hhi_excludes_unmapped():
    fd, path_str = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    path = Path(path_str)

    conn = create_db(path)
    cur = conn.cursor()
    # project 1, year 2020
    cur.execute("INSERT INTO Organisation VALUES (?, ?)", (1, "OrgA"))
    # author 10 affiliated with OrgA, author 11 unmapped
    # Affiliation schema is (organisation_id, person_id)
    cur.execute("INSERT INTO Affiliation VALUES (?, ?)", (1, 10))
    # two proposals with single authors
    cur.execute(
        "INSERT INTO ProposalRevision VALUES (1, 100, 0, '2020-01-01T00:00:00Z')"
    )
    cur.execute("INSERT INTO ProposalRevisionAuthor VALUES (1, 100, 0, 10)")
    cur.execute(
        "INSERT INTO ProposalRevision VALUES (1, 101, 0, '2020-02-01T00:00:00Z')"
    )
    cur.execute("INSERT INTO ProposalRevisionAuthor VALUES (1, 101, 0, 11)")
    conn.commit()
    conn.close()

    hhi = governance.compute_independence_hhi_per_year(path, 1)
    # only OrgA counted -> HHI = 1.0
    assert hhi.get(2020) == pytest.approx(1.0)
    os.remove(path)


def test_betweenness_centralization_path_graph():
    fd, path_str = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    path = Path(path_str)

    conn = create_db(path)
    cur = conn.cursor()
    # Create three authors connected as 1-2 and 2-3 via revisions
    cur.execute(
        "INSERT INTO ProposalRevision VALUES (1, 200, 0, '2021-01-01T00:00:00Z')"
    )
    cur.execute("INSERT INTO ProposalRevisionAuthor VALUES (1, 200, 0, 1)")
    cur.execute("INSERT INTO ProposalRevisionAuthor VALUES (1, 200, 0, 2)")
    cur.execute(
        "INSERT INTO ProposalRevision VALUES (1, 201, 0, '2021-02-01T00:00:00Z')"
    )
    cur.execute("INSERT INTO ProposalRevisionAuthor VALUES (1, 201, 0, 2)")
    cur.execute("INSERT INTO ProposalRevisionAuthor VALUES (1, 201, 0, 3)")
    conn.commit()

    centralization = governance.compute_betweenness_centralization_per_year(path, 1)
    # path of 3 nodes should have centralization 1.0 for that year
    assert centralization.get(2021) == pytest.approx(1.0)
    conn.close()
    os.remove(path)


def test_newcomer_onboarding_rate():
    fd, path_str = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    path = Path(path_str)

    conn = create_db(path)
    cur = conn.cursor()
    # author 20 had prior activity in 2019
    cur.execute(
        "INSERT INTO ProposalRevision VALUES (1, 300, 0, '2019-01-01T00:00:00Z')"
    )
    cur.execute("INSERT INTO ProposalRevisionAuthor VALUES (1, 300, 0, 20)")
    # in 2022, two proposals: one by newcomer 21, one by returning 20
    cur.execute(
        "INSERT INTO ProposalRevision VALUES (1, 301, 0, '2022-01-01T00:00:00Z')"
    )
    cur.execute("INSERT INTO ProposalRevisionAuthor VALUES (1, 301, 0, 21)")
    cur.execute(
        "INSERT INTO ProposalRevision VALUES (1, 302, 0, '2022-02-01T00:00:00Z')"
    )
    cur.execute("INSERT INTO ProposalRevisionAuthor VALUES (1, 302, 0, 20)")
    conn.commit()

    rate = governance.compute_newcomer_success_rate_per_year(path, 1)
    # one of two proposals in 2022 was by a newcomer -> 0.5
    assert rate.get(2022) == pytest.approx(0.5)
    conn.close()
    os.remove(path)
