import json
from collections import defaultdict
from pathlib import Path

import polars as pl

from dataloader import TABLE_SCHEMAS, IndividualProjectContext


def merge_duplicate_companies_in_contexts(
    contexts: list[IndividualProjectContext], output_dir: Path
) -> list[IndividualProjectContext]:
    alias_filepath = "data/companies_aliases.json"
    alias_dict: dict[str, str] = {}

    # 1. Load and flatten aliases into a clean dictionary
    with open(alias_filepath, "r", encoding="utf-8") as f:
        for master, aliases in json.load(f).items():
            alias_dict[master.lower()] = master.lower()  # Force lowercase master
            for alias in aliases:
                alias_dict[alias.lower()] = master.lower()

    # Verify and immediately update chained aliases to the ultimate master
    for alias, master in list(alias_dict.items()):
        if master in alias_dict and alias_dict[master] != master:
            ultimate_master = alias_dict[master]
            print(
                f"Warning: Master name '{master}' is an alias for '{ultimate_master}'. Updating map."
            )

            # Immediate shortcut mapping
            alias_dict[alias] = ultimate_master

    # 2. Reconstruct into the original format, deduplicate, and sort alphabetically
    updated_json_data = defaultdict(set)
    for alias, master in alias_dict.items():
        if alias != master:  # Don't add the master to its own alias list
            updated_json_data[master].add(alias)
        else:
            # Ensure the master key exists even if it has no aliases
            _ = updated_json_data[master]

    # Format with sorted keys and sorted, unique alias lists
    final_output = {
        master: sorted(list(aliases))
        for master, aliases in sorted(updated_json_data.items())
    }

    all_clean_companies = set()
    all_original_companies = set()
    # Save back to the original file
    with open(alias_filepath, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    for ctx in contexts:
        if ctx.organisations.is_empty():
            continue

        all_original_companies.update(ctx.organisations["organisation_name"].to_list())

        # 2. Clean the names strictly (lowercase and spaces only)
        normalized = ctx.organisations.with_columns(
            pl.col("organisation_name")
            .str.to_lowercase()
            .str.replace_all(r"\b(inc|llc|ltd|corp|corporation|gmbh|sa)\b\.?", "")
            .str.replace_all(
                r"[^a-z ]", " "
            )  # Replaces symbols like '@' and '-' with spaces
            .str.replace_all(r"\s+", " ")  # Collapses multiple spaces into one
            .str.strip_chars()
            .alias("norm_key")
        )

        # Apply alias mapping if available
        if alias_dict:
            normalized = normalized.with_columns(
                pl.col("norm_key").replace(alias_dict, default=pl.col("norm_key"))
            )

        # --- REMOVE EMPTY COMPANY NAMES ---
        # Filters out strings that became empty after cleaning/normalization
        normalized = normalized.filter(pl.col("norm_key") != "")

        # 3. Create a master mapping table using the normalized keys
        master_map = normalized.group_by("norm_key").agg(
            pl.col("organisation_id").min().alias("master_id"),
            pl.col("norm_key")
            .first()
            .alias("master_name"),  # FIX: Save the normalized name here
        )

        # Merge mapping back
        resolved = normalized.join(master_map, on="norm_key", how="left").with_columns(
            pl.col("master_id").fill_null(pl.col("organisation_id")),
            pl.col("master_name").fill_null(pl.col("norm_key")),
        )

        # 4. Update Organisations Table
        clean_orgs = (
            resolved.select(
                pl.col("master_id").alias("organisation_id"),
                pl.col("master_name").alias("organisation_name"),
            )
            .unique(subset=["organisation_id"])
            .select(list(TABLE_SCHEMAS["Organisation"].keys()))
        )

        # 5. Update Affiliations Table
        if not ctx.affiliations.is_empty():
            id_bridge = resolved.select(
                ["organisation_id", "master_id", "norm_key"]
            ).unique()

            # Base join to align old IDs with their resolved master details
            updated_affiliations = ctx.affiliations.join(
                id_bridge, on="organisation_id", how="left"
            ).filter(pl.col("master_id").is_not_null())

            # --- OPENJDK ORACLE FILTER ---
            is_openjdk = ctx.project_id == 6
            if is_openjdk:
                # Drop affiliations linking to oracle (post alias-mapping evaluation)
                updated_affiliations = updated_affiliations.filter(
                    pl.col("norm_key") != "oracle"
                )

            ctx.affiliations = (
                updated_affiliations.with_columns(
                    pl.col("master_id").alias("organisation_id")
                )
                .drop(["master_id", "norm_key"])
                .unique()
                .select(list(TABLE_SCHEMAS["Affiliation"].keys()))
            )
        ctx.organisations = clean_orgs
        if not clean_orgs.is_empty():
            all_clean_companies.update(clean_orgs["organisation_name"].to_list())

    original_count = len(all_original_companies)
    clean_count = len(all_clean_companies)
    print(
        f"Deduplication complete: {original_count} original companies reduced to {clean_count} unique companies."
    )

    # 6. Save text outputs
    for filename, dataset in [
        ("companies_original.txt", all_original_companies),
        ("companies.txt", all_clean_companies),
    ]:
        if dataset:
            sorted_data = sorted(
                [str(name).strip() for name in dataset if str(name).strip()]
            )
            with open(output_dir / filename, "w", encoding="utf-8") as f:
                f.write("\n".join(sorted_data) + "\n")

    return contexts
