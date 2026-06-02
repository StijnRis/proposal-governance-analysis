import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence

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


# Explicit container type for intermediate step
class _MergedProjectData(NamedTuple):
    project: pl.DataFrame
    proposals: pl.DataFrame
    proposal_revisions: pl.DataFrame
    proposal_revision_authors: pl.DataFrame
    proposal_status: pl.DataFrame
    comments: pl.DataFrame
    persons: pl.DataFrame
    person_identifiers: pl.DataFrame
    organisations: pl.DataFrame
    affiliations: pl.DataFrame
    related_proposals: pl.DataFrame


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


def _slice_data_by_project(
    project_data: _MergedProjectData,
    max_proposals: int | None = None,
    seed: int | None = 42,  # Added a seed for reproducible test runs
) -> list[IndividualProjectContext]:
    """Traces relational tables back to a root project_id and splits them into clean contexts."""

    base_proposals = project_data.proposals

    # 1. Randomly sample up to max_proposals *per project*
    if max_proposals is not None:
        base_proposals = (
            base_proposals
            # Shuffle the entire dataframe randomly
            .sample(fraction=1.0, shuffle=True, seed=seed)
            # Group by project and take the first N sliced rows of that group
            .group_by("project_id", maintain_order=True)
            .head(max_proposals)
        )

    # 2. Get the unique projects that made the cut
    allowed_project_ids = base_proposals.select("project_id").unique()
    unique_projects = (
        project_data.project.join(allowed_project_ids, on="project_id", how="inner")
        .select(["project_id", "project_name"])
        .unique()
    )

    contexts = []

    for row in unique_projects.iter_rows(named=True):
        p_id, p_name = row["project_id"], row["project_name"]

        # Project table filter
        proj = project_data.project.filter(pl.col("project_id") == p_id)

        # Proposal filtering tied strictly to our randomly sampled subset
        proposals = base_proposals.filter(pl.col("project_id") == p_id)

        # Get by project id
        revisions = project_data.proposal_revisions.filter(pl.col("project_id") == p_id)
        revision_authors = project_data.proposal_revision_authors.filter(
            pl.col("project_id") == p_id
        )
        comments = project_data.comments.filter(pl.col("project_id") == p_id)
        status = project_data.proposal_status.filter(pl.col("project_id") == p_id)
        related_proposals = project_data.related_proposals.filter(
            pl.col("project_id") == p_id
        )

        # Get by person id
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
        people = project_data.persons.join(
            project_people_ids, on="person_id", how="inner"
        )
        person_idents = project_data.person_identifiers.join(
            project_people_ids, on="person_id", how="inner"
        )
        affils = project_data.affiliations.join(
            project_people_ids, on="person_id", how="inner"
        )

        # Get by organisation id
        project_organisation_ids = (
            affils.select(pl.col("organisation_id").alias("organisation_id"))
            .drop_nulls()
            .unique()
        )
        orgs = project_data.organisations.join(
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

    contexts = sorted(contexts, key=lambda c: c.project_id)
    return contexts


def load_all_projects(
    db_paths: Sequence[Path], max_proposals: int | None
) -> list[IndividualProjectContext]:
    """Reads multiple matching SQLite databases, merges data, and slices into isolated project contexts."""
    if not db_paths:
        raise ValueError("Must provide at least one database path.")

    tables = {k: [] for k in TABLE_SCHEMAS.keys()}

    # Extract raw dataframes sequentially from each file
    for path in db_paths:
        conn = sqlite3.connect(str(path))
        try:
            for table_name in tables.keys():
                df = _safe_read_table(conn, table_name)
                tables[table_name].append(df)
        finally:
            conn.close()

    # Compile the database matrix diagonally
    merged_data = _MergedProjectData(
        project=pl.concat(tables["Project"], how="diagonal"),
        proposals=pl.concat(tables["Proposal"], how="diagonal"),
        proposal_revisions=pl.concat(tables["ProposalRevision"], how="diagonal"),
        proposal_revision_authors=pl.concat(
            tables["ProposalRevisionAuthor"], how="diagonal"
        ),
        proposal_status=pl.concat(tables["ProposalStatus"], how="diagonal"),
        comments=pl.concat(tables["Comment"], how="diagonal"),
        persons=pl.concat(tables["Person"], how="diagonal"),
        person_identifiers=pl.concat(tables["PersonIdentifier"], how="diagonal"),
        organisations=pl.concat(tables["Organisation"], how="diagonal"),
        affiliations=pl.concat(tables["Affiliation"], how="diagonal"),
        related_proposals=pl.concat(tables["RelatedProposal"], how="diagonal"),
    )

    # Return structured isolated array
    return _slice_data_by_project(merged_data, max_proposals=max_proposals)
