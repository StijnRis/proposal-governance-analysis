import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import polars as pl


# --- Schemas ---
class ProjectSchema:
    project_id: pl.Int64
    project_name: pl.String
    enhancement_proposal_name: pl.String
    copyright: pl.String


class PersonSchema:
    person_id: pl.Int64
    full_name: pl.String


class PersonIdentifierSchema:
    person_id: pl.Int64
    domain: pl.String
    identifier_type: pl.String
    identifier: pl.String


class OrganisationSchema:
    organisation_id: pl.Int64
    organisation_name: pl.String


class AffiliationSchema:
    organisation_id: pl.Int64
    person_id: pl.Int64


class ProposalSchema:
    project_id: pl.Int64
    proposal_id: pl.String
    topic: pl.String
    proposal_type: pl.String


class ProposalStatusSchema:
    project_id: pl.Int64
    proposal_id: pl.String
    status_index: pl.Int64
    raw_status: pl.String
    normalised_status: pl.String
    created_at: pl.Datetime


class ProposalRevisionSchema:
    project_id: pl.Int64
    proposal_id: pl.String
    revision_index: pl.Int64
    title: pl.String
    created_at: pl.Datetime
    content: pl.String
    implemented_at_version: pl.String


class ProposalRevisionAuthorSchema:
    project_id: pl.Int64
    proposal_id: pl.String
    revision_index: pl.Int64
    author_id: pl.Int64


class RelatedProposalSchema:
    project_id: pl.Int64
    proposal_id: pl.String
    related_project_id: pl.Int64
    related_proposal_id: pl.String
    type: pl.String


class CommentSchema:
    comment_id: pl.Int64
    author_id: pl.Int64
    project_id: pl.Int64
    proposal_id: pl.String
    comment_on_comment_id: pl.Int64
    created_at: pl.Datetime
    content: pl.String


@dataclass
class IndividualProjectContext:
    project_id: int
    project_name: str
    project: pl.DataFrame
    proposals: pl.DataFrame
    proposal_revisions: pl.DataFrame
    proposal_revision_authors: pl.DataFrame
    proposal_statuses: pl.DataFrame
    comments: pl.DataFrame
    persons: pl.DataFrame
    person_identifiers: pl.DataFrame
    organisations: pl.DataFrame
    affiliations: pl.DataFrame
    related_proposals: pl.DataFrame


TABLE_SCHEMAS = {
    "Project": {
        "project_id": pl.Int64,
        "project_name": pl.String,
        "enhancement_proposal_name": pl.String,
        "copyright": pl.String,
    },
    "Person": {
        "person_id": pl.Int64,
        "full_name": pl.String,
    },
    "PersonIdentifier": {
        "person_id": pl.Int64,
        "domain": pl.String,
        "identifier_type": pl.String,
        "identifier": pl.String,
    },
    "Organisation": {
        "organisation_id": pl.Int64,
        "organisation_name": pl.String,
    },
    "Affiliation": {
        "organisation_id": pl.Int64,
        "person_id": pl.Int64,
    },
    "Proposal": {
        "project_id": pl.Int64,
        "proposal_id": pl.String,
        "topic": pl.String,
        "proposal_type": pl.String,
    },
    "ProposalStatus": {
        "project_id": pl.Int64,
        "proposal_id": pl.String,
        "status_index": pl.Int64,
        "raw_status": pl.String,
        "normalised_status": pl.String,
        "created_at": pl.Datetime("ns", "UTC"),
    },
    "ProposalRevision": {
        "project_id": pl.Int64,
        "proposal_id": pl.String,
        "revision_index": pl.Int64,
        "title": pl.String,
        "created_at": pl.Datetime("ns", "UTC"),
        "content": pl.String,
        "implemented_at_version": pl.String,
    },
    "ProposalRevisionAuthor": {
        "project_id": pl.Int64,
        "proposal_id": pl.String,
        "revision_index": pl.Int64,
        "author_id": pl.Int64,
    },
    "RelatedProposal": {
        "project_id": pl.Int64,
        "proposal_id": pl.String,
        "related_project_id": pl.Int64,
        "related_proposal_id": pl.String,
        "type": pl.String,
    },
    "Comment": {
        "comment_id": pl.Int64,
        "author_id": pl.Int64,
        "project_id": pl.Int64,
        "proposal_id": pl.String,
        "comment_on_comment_id": pl.Int64,
        "created_at": pl.Datetime("ns", "UTC"),
        "content": pl.String,
    },
}


def _parse_flexible_datetime(df: pl.DataFrame, col_name: str) -> pl.DataFrame:
    """Parses a column into pl.Datetime forcing standard UTC format."""
    if col_name not in df.columns:
        return df

    dtype = df.schema[col_name]

    if dtype in (pl.String, pl.Object):
        return df.with_columns(
            pl.col(col_name)
            .str.to_datetime(strict=False, time_zone="UTC")
            .alias(col_name)
        )

    if dtype.is_numeric():
        return df.with_columns(
            pl.when(pl.col(col_name) > 1e14)
            .then(pl.col(col_name).cast(pl.Int64).cast(pl.Datetime("ns")))
            .when(pl.col(col_name) > 1e11)
            .then(pl.col(col_name).cast(pl.Int64).cast(pl.Datetime("ms")))
            .otherwise(pl.col(col_name).cast(pl.Int64).cast(pl.Datetime("s")))
            .alias(col_name)
        ).with_columns(pl.col(col_name).dt.replace_time_zone("UTC"))

    return df


def _safe_read_table(conn: sqlite3.Connection, table_name: str) -> pl.DataFrame:
    """Safely reads a table from SQLite, parsing and enforcing strict types per schema."""
    df = pl.read_database(
        f"SELECT * FROM {table_name}",
        connection=conn,
        infer_schema_length=10000,
    )

    if "created_at" in df.columns:
        df = _parse_flexible_datetime(df, "created_at")

    if table_name in TABLE_SCHEMAS:
        schema = TABLE_SCHEMAS[table_name]

        casts = []
        for col_name, expected_type in schema.items():
            if col_name in df.columns:
                if isinstance(expected_type, pl.Datetime) and isinstance(
                    df.schema[col_name], pl.Datetime
                ):
                    continue
                casts.append(pl.col(col_name).cast(expected_type, strict=False))

        if casts:
            df = df.with_columns(casts)

        df = df.select([pl.col(c) for c in schema.keys() if c in df.columns])

    return df


def _extract_contexts_from_db(
    conn: sqlite3.Connection,
    max_proposals: int | None = None,
    seed: int | None = 42,
) -> list[IndividualProjectContext]:
    """Extracts and slices individual project contexts out of a single database connection."""

    # 1. Read the necessary driving tables first
    raw_proposals = _safe_read_table(conn, "Proposal")
    if raw_proposals.is_empty():
        return []

    # Apply sampling per project directly within this DB's scope
    base_proposals = raw_proposals
    if max_proposals is not None:
        base_proposals = (
            base_proposals.sample(fraction=1.0, shuffle=True, seed=seed)
            .group_by("project_id", maintain_order=True)
            .head(max_proposals)
        )

    allowed_project_ids = base_proposals.select("project_id").unique()

    raw_projects = _safe_read_table(conn, "Project")
    unique_projects = (
        raw_projects.join(allowed_project_ids, on="project_id", how="inner")
        .select(["project_id", "project_name"])
        .unique()
    )

    if unique_projects.is_empty():
        return []

    # 2. Lazy load remaining tables for this database
    raw_revisions = _safe_read_table(conn, "ProposalRevision")
    raw_revision_authors = _safe_read_table(conn, "ProposalRevisionAuthor")
    raw_comments = _safe_read_table(conn, "Comment")
    raw_status = _safe_read_table(conn, "ProposalStatus")
    raw_related_proposals = _safe_read_table(conn, "RelatedProposal")

    raw_persons = _safe_read_table(conn, "Person")
    raw_person_identifiers = _safe_read_table(conn, "PersonIdentifier")
    raw_affiliations = _safe_read_table(conn, "Affiliation")
    raw_organisations = _safe_read_table(conn, "Organisation")

    contexts = []

    # 3. Slice tables strictly belonging to the projects found inside this database
    for row in unique_projects.iter_rows(named=True):
        p_id, p_name = row["project_id"], row["project_name"]

        proj = raw_projects.filter(pl.col("project_id") == p_id)
        proposals = base_proposals.filter(pl.col("project_id") == p_id)
        revisions = raw_revisions.filter(pl.col("project_id") == p_id)
        revision_authors = raw_revision_authors.filter(pl.col("project_id") == p_id)
        comments = raw_comments.filter(pl.col("project_id") == p_id)
        status = raw_status.filter(pl.col("project_id") == p_id)
        related_proposals = raw_related_proposals.filter(pl.col("project_id") == p_id)

        # Trace dynamic relationships (People)
        project_people_ids = (
            pl.concat(
                [
                    revision_authors.select(pl.col("author_id").alias("person_id")),
                    comments.select(pl.col("author_id").alias("person_id")),
                ]
            )
            .drop_nulls()
            .unique()
        )

        people = raw_persons.join(project_people_ids, on="person_id", how="inner")
        person_idents = raw_person_identifiers.join(
            project_people_ids, on="person_id", how="inner"
        )
        affils = raw_affiliations.join(project_people_ids, on="person_id", how="inner")

        # Trace dynamic relationships (Organisations)
        project_organisation_ids = (
            affils.select(pl.col("organisation_id")).drop_nulls().unique()
        )
        orgs = raw_organisations.join(
            project_organisation_ids, on="organisation_id", how="inner"
        )

        contexts.append(
            IndividualProjectContext(
                project_id=p_id,
                project_name=p_name,
                project=proj,
                proposals=proposals,
                proposal_revisions=revisions,
                proposal_revision_authors=revision_authors,
                proposal_statuses=status,
                comments=comments,
                persons=people,
                person_identifiers=person_idents,
                organisations=orgs,
                affiliations=affils,
                related_proposals=related_proposals,
            )
        )

    return contexts


def load_all_projects(
    db_paths: Sequence[Path], max_proposals: int | None, seed: int | None = 42
) -> list[IndividualProjectContext]:
    """Reads multiple matching SQLite databases sequentially and constructs isolated project contexts directly."""
    if not db_paths:
        raise ValueError("Must provide at least one database path.")

    all_contexts = []

    # debug = os.getenv("DEBUG", "False").lower() == "true"

    # Extract clean contexts from each file individually
    for path in db_paths:
        print(f"Processing database: {path}")
        conn = sqlite3.connect(str(path))
        try:
            db_contexts = _extract_contexts_from_db(
                conn, max_proposals=max_proposals, seed=seed
            )
            all_contexts.extend(db_contexts)
        finally:
            conn.close()

        # if debug:
        #     print("DEBUG: only load one project")
        #     break

    # Final sort across all extracted projects
    return sorted(all_contexts, key=lambda c: c.project_name.lower())
