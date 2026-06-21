import os
import sqlite3
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Sequence
import polars as pl

from dataloader import IndividualProjectContext, _safe_read_table

# [Keep your existing schemas, TABLE_SCHEMAS, _parse_flexible_datetime, 
# _safe_read_table, and load_all_projects implementation here exactly as they are]


def get_random_proposals_with_metadata(
    db_paths: Sequence[Path], 
    num_proposals: int = 10, 
    seed: int = 42
) -> list[IndividualProjectContext]:
    """
    Selects N random proposals across all databases, extracts all their linked 
    metadata, and sorts the returned results by proposal_id.
    """
    if not db_paths:
        raise ValueError("Must provide at least one database path.")

    # 1. Read raw Proposal and Project tables globally first to make a global random selection
    all_proposals = []
    for path in db_paths:
        with sqlite3.connect(str(path)) as conn:
            df_prop = _safe_read_table(conn, "Proposal")
            if not df_prop.is_empty():
                all_proposals.append(df_prop)

    if not all_proposals:
        print("No proposals found in the database(s).")
        return []

    global_proposals = pl.concat(all_proposals)

    # 2. Randomly sample N proposals globally using the fixed seed
    sampled_proposals = (
        global_proposals
        .sort("proposal_id")
        .sample(n=num_proposals, shuffle=True, seed=seed)
    )

    # Grab unique pairings of project_id and proposal_id to cleanly filter other tables
    target_ids = sampled_proposals.select(["project_id", "proposal_id"]).unique()
    target_project_ids = sampled_proposals.select("project_id").unique()

    # 3. Reload databases, slicing metadata dynamically to match only our sampled elements
    contexts = []
    for path in db_paths:
        with sqlite3.connect(str(path)) as conn:
            # Check if this database contains any of our sampled projects
            raw_projects = _safe_read_table(conn, "Project")
            db_projects = raw_projects.join(target_project_ids, on="project_id", how="inner")
            if db_projects.is_empty():
                continue

            # Load matching, trimmed-down downstream metadata tables
            proposals = sampled_proposals.join(db_projects.select("project_id"), on="project_id", how="inner")
            
            revisions = _safe_read_table(conn, "ProposalRevision").join(target_ids, on=["project_id", "proposal_id"], how="inner").sort("proposal_id")
            revision_authors = _safe_read_table(conn, "ProposalRevisionAuthor").join(target_ids, on=["project_id", "proposal_id"], how="inner").sort("proposal_id")
            comments = _safe_read_table(conn, "Comment").join(target_ids, on=["project_id", "proposal_id"], how="inner").sort("proposal_id")
            statuses = _safe_read_table(conn, "ProposalStatus").join(target_ids, on=["project_id", "proposal_id"], how="inner").sort("proposal_id")
            related_proposals = _safe_read_table(conn, "RelatedProposal").join(target_ids, on=["project_id", "proposal_id"], how="inner").sort("proposal_id")

            # Extract specific people linked strictly to these 10 proposals
            people_ids = pl.concat([
                revision_authors.select(pl.col("author_id").alias("person_id")),
                comments.select(pl.col("author_id").alias("person_id")),
            ]).drop_nulls().unique()

            persons = _safe_read_table(conn, "Person").join(people_ids, on="person_id", how="inner")
            person_identifiers = _safe_read_table(conn, "PersonIdentifier").join(people_ids, on="person_id", how="inner")
            affiliations = _safe_read_table(conn, "Affiliation").join(people_ids, on="person_id", how="inner")
            
            org_ids = affiliations.select("organisation_id").drop_nulls().unique()
            organisations = _safe_read_table(conn, "Organisation").join(org_ids, on="organisation_id", how="inner")

            # Construct our contexts per project group
            for row in db_projects.unique(subset=["project_id"]).iter_rows(named=True):
                p_id, p_name = row["project_id"], row["project_name"]
                
                contexts.append(
                    IndividualProjectContext(
                        project_id=p_id,
                        project_name=p_name,
                        project=db_projects.filter(pl.col("project_id") == p_id),
                        proposals=proposals.filter(pl.col("project_id") == p_id),
                        proposal_revisions=revisions.filter(pl.col("project_id") == p_id),
                        proposal_revision_authors=revision_authors.filter(pl.col("project_id") == p_id),
                        proposal_statuses=statuses.filter(pl.col("project_id") == p_id),
                        comments=comments.filter(pl.col("project_id") == p_id),
                        persons=persons,
                        person_identifiers=person_identifiers,
                        organisations=organisations,
                        affiliations=affiliations,
                        related_proposals=related_proposals.filter(pl.col("project_id") == p_id),
                    )
                )

    return sorted(contexts, key=lambda c: c.project_name.lower())

def print_project_contexts(contexts: list[IndividualProjectContext]):
    """Pretty prints the contents of our isolated contexts, showing all rows."""
    print(f"\n{'='*24} SAMPLED METADATA RESULTS {'='*24}")
    
    # Configure Polars to show unlimited rows while leaving column widths dynamic
    with pl.Config(tbl_rows=-1):
        for ctx in contexts:
            print(f"\n🔹 Project: {ctx.project_name} (ID: {ctx.project_id})")
            print(f"{'-'*60}")
            
            for field in fields(ctx):
                # Skip ID and Name attributes since we printed them in the header
                if field.name in ["project_id", "project_name"]:
                    continue
                    
                df: pl.DataFrame = getattr(ctx, field.name)
                print(f" 📂 Dataframe: {field.name} (Rows: {df.height})")
                if not df.is_empty():
                    print(df)
                print()


# --- Execution Block ---
if __name__ == "__main__":
    # Replace with the actual path to your SQLite database file(s)
    DATABASE_PATHS = [Path(r"C:\Users\risse\University\CSE3000 Research Project\proposal-governance-analysis\data\proposals\cplusplus_proposals_2026-06-02.sqlite3")]
    
    # Verify file exists before running example logic
    if not any(p.exists() for p in DATABASE_PATHS):
        print("Please configure valid database path strings in the script execution block.")
    else:
        sampled_contexts = get_random_proposals_with_metadata(
            db_paths=DATABASE_PATHS, 
            num_proposals=10, 
            seed=42
        )
        print_project_contexts(sampled_contexts)