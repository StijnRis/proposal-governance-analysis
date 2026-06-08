from typing import List

import polars as pl

from dataloader import IndividualProjectContext


def merge_duplicate_people(
    contexts: List[IndividualProjectContext],
) -> List[IndividualProjectContext]:
    """
    Merges all persons across all contexts that share:
    1. The exact same full_name (or are both NULL).
    2. The same identifier payload based on rules:
       - If identifier_type is 'email', domain can be different.
       - For other types, both domain and identifier_type must match.

    Updates all foreign key references across the dataframes inside each context.
    """
    if not contexts:
        print("No contexts provided.")
        return contexts

    # --- 1. Consolidate Global Views of Person and PersonIdentifier ---
    global_persons = pl.concat([c.persons for c in contexts]).unique(
        subset=["person_id"]
    )
    global_identifiers = pl.concat([c.person_identifiers for c in contexts]).unique()

    if global_persons.is_empty() or global_identifiers.is_empty():
        print("Person or Identifier tables are empty. No merges performed.")
        return contexts

    # --- 2. Separate Identifiers by Logic Types ---
    emails = global_identifiers.filter(pl.col("identifier_type") == "email")
    others = global_identifiers.filter(pl.col("identifier_type") != "email")

    # Rule A: Emails match on type + identifier (domain can vary)
    email_overlaps = (
        emails.join(emails, on=["identifier_type", "identifier"], suffix="_other")
        .filter(pl.col("person_id") != pl.col("person_id_other"))
        .select(["person_id", "person_id_other"])
    )

    # Rule B: Non-emails match strictly on domain + type + identifier
    other_overlaps = (
        others.join(
            others, on=["domain", "identifier_type", "identifier"], suffix="_other"
        )
        .filter(pl.col("person_id") != pl.col("person_id_other"))
        .select(["person_id", "person_id_other"])
    )

    # Combine both rule outputs together
    id_overlaps = pl.concat([email_overlaps, other_overlaps]).unique()

    if id_overlaps.is_empty():
        print("Merged 0 duplicate records (0 distinct people affected).")
        return contexts

    # --- 3. Filter Overlaps by Name Consistency (Same Name or Both NULL) ---
    name_checks = id_overlaps.join(global_persons, on="person_id", how="inner").join(
        global_persons,
        left_on="person_id_other",
        right_on="person_id",
        how="inner",
        suffix="_other",
    )

    duplicate_pairs = name_checks.filter(
        (pl.col("full_name") == pl.col("full_name_other"))
        | (pl.col("full_name").is_null() & pl.col("full_name_other").is_null())
    ).select(["person_id", "person_id_other"])

    if duplicate_pairs.is_empty():
        print("Merged 0 duplicate records (0 distinct people affected).")
        return contexts

    # --- 4. Resolve Groups Into an ID Mapping Dictionary (Graph Components) ---
    adj_list = {}
    for row in duplicate_pairs.iter_rows():
        u, v = row[0], row[1]
        adj_list.setdefault(u, set()).add(v)
        adj_list.setdefault(v, set()).add(u)

    id_mapping = {}
    visited = set()
    distinct_people_merged_count = 0

    for node in adj_list:
        if node not in visited:
            component = set()
            queue = [node]
            while queue:
                curr = queue.pop(0)
                if curr not in component:
                    component.add(curr)
                    queue.extend(adj_list[curr])

            visited.update(component)
            survivor_id = min(component)
            for old_id in component:
                id_mapping[old_id] = survivor_id

            distinct_people_merged_count += 1

    total_records_removed = len(id_mapping) - distinct_people_merged_count
    print(
        f"Merged {total_records_removed} duplicate records across {distinct_people_merged_count} distinct people."
    )

    # Convert mapping to a Polars DataFrame for joining
    mapping_df = pl.DataFrame(
        {"old_id": list(id_mapping.keys()), "new_id": list(id_mapping.values())}
    )

    def remap_column(df: pl.DataFrame, col_name: str) -> pl.DataFrame:
        """Helper to replace old IDs with new IDs in a specific column."""
        if col_name not in df.columns or df.is_empty():
            return df
        return (
            df.join(mapping_df, left_on=col_name, right_on="old_id", how="left")
            .with_columns(pl.coalesce(["new_id", col_name]).alias(col_name))
            .drop("new_id")
        )

    # --- 5. Update and Deduplicate Datasets Inside Contexts ---
    updated_contexts = []
    for ctx in contexts:
        persons = remap_column(ctx.persons, "person_id").unique(subset=["person_id"])
        person_identifiers = remap_column(ctx.person_identifiers, "person_id").unique()
        affiliations = remap_column(
            ctx.affiliations, ["organisation_id", "person_id"]
        ).unique()

        proposal_revision_authors = remap_column(
            ctx.proposal_revision_authors, "author_id"
        ).unique()
        comments = remap_column(ctx.comments, "author_id")

        updated_contexts.append(
            IndividualProjectContext(
                project_id=ctx.project_id,
                project_name=ctx.project_name,
                project=ctx.project,
                proposals=ctx.proposals,
                proposal_revisions=ctx.proposal_revisions,
                proposal_revision_authors=proposal_revision_authors,
                proposal_statuses=ctx.proposal_statuses,
                comments=comments,
                persons=persons,
                person_identifiers=person_identifiers,
                organisations=ctx.organisations,
                affiliations=affiliations,
                related_proposals=ctx.related_proposals,
            )
        )

    return updated_contexts
