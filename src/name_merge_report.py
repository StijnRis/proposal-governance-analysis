import sqlite3
from pathlib import Path

import polars as pl

from enhance_data.merge_people import build_person_id_mapping

ACTIVE_PEOPLE_IDS = """
SELECT author_id FROM ProposalRevisionAuthor
UNION SELECT author_id FROM Comment
"""
SCHEMA_OVERRIDES = {
    "person_id": pl.Int64,
    "full_name": pl.String,
    "domain": pl.String,
    "identifier_type": pl.String,
    "identifier": pl.String,
}


def read_active_table(db_path: Path, table: str) -> pl.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pl.read_database(
            f"SELECT * FROM {table} WHERE person_id IN ({ACTIVE_PEOPLE_IDS})",
            connection=conn,
            schema_overrides=SCHEMA_OVERRIDES,
        )


def read_all_active(db_paths: list[Path], table: str) -> pl.DataFrame:
    frames = [read_active_table(path, table) for path in db_paths]
    return pl.concat([f.select(frames[0].columns) for f in frames]).unique()


def count_merged_names(
    persons: pl.DataFrame, id_mapping: dict[int, int]
) -> pl.DataFrame:
    return (
        persons.filter(pl.col("person_id").is_in(list(id_mapping)))
        .group_by("full_name")
        .agg(pl.len().alias("count"))
        .rename({"full_name": "name"})
        .sort(["count", "name"], descending=[True, False], nulls_last=True)
    )


def main() -> None:
    db_paths = sorted(
        path
        for ext in ("*.sqlite3", "*.db", "*.sqlite")
        for path in Path("data/proposals").glob(ext)
    )
    persons = read_all_active(db_paths, "Person").unique(subset=["person_id"])
    identifiers = read_all_active(db_paths, "PersonIdentifier")
    id_mapping = build_person_id_mapping(persons, identifiers)
    merged = count_merged_names(persons, id_mapping)
    output_path = Path("output/name_merge_counts.csv")
    output_path.parent.mkdir(exist_ok=True)
    merged.write_csv(output_path)
    print(
        f"{len(id_mapping) - len(set(id_mapping.values()))} records removed, "
        f"{len(set(id_mapping.values()))} people kept -> {output_path}"
    )


if __name__ == "__main__":
    main()
