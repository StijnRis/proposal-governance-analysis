import sqlite3
from pathlib import Path
from typing import Dict, List


def attach_and_create_union_views(
    conn: sqlite3.Connection, schema_sql_path: Path, db_paths: List[Path]
) -> List[Path]:
    """Attach the given database files to `conn` and create TEMP VIEWs that
    UNION ALL each table across the attached databases.

    Returns the list of attached database Paths (those that were attached). It
    assumes the schema file is the canonical schema to derive table names.
    """
    # load ideal schema to determine table names
    ideal_conn = sqlite3.connect(":memory:")
    try:
        with schema_sql_path.open("r", encoding="utf-8") as f:
            ideal_conn.executescript(f.read())
        ideal_schema = get_schema_dict(ideal_conn)
    finally:
        ideal_conn.close()

    attached: List[Path] = []
    aliases: List[str] = []
    # Attach each provided DB with alias d0, d1, ...
    for idx, p in enumerate(db_paths):
        alias = f"d{idx}"
        # sanitize single quotes in path
        pstr = str(p).replace("'", "''")
        conn.execute(f"ATTACH DATABASE '{pstr}' AS {alias}")
        attached.append(p)
        aliases.append(alias)

    # For each table in the ideal schema, create a TEMP VIEW that unions across
    # all attached DB aliases. Include the 'main' database (the connection's
    # primary DB) as well.
    for table in ideal_schema.keys():
        parts = [f"SELECT * FROM main.'{table}'"]
        for a in aliases:
            parts.append(f"SELECT * FROM {a}.'{table}'")
        union_sql = " UNION ALL ".join(parts)
        view_sql = f"CREATE TEMP VIEW IF NOT EXISTS '{table}' AS {union_sql};"
        conn.execute(view_sql)

    return attached


def get_schema_dict(conn: sqlite3.Connection) -> Dict[str, str]:
    """Extracts table names and their exact CREATE SQL from a database connection."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    return {row[0]: row[1] for row in cursor.fetchall()}


def verify_schema(schema_sql_path: Path, db_path: Path) -> List[str]:
    """Verify that the SQLite database at `db_path` matches the schema in `schema_sql_path`.

    Returns a list of human-readable mismatch/error strings. Empty list means the
    schemas match.
    """
    schema_sql_path = Path(schema_sql_path)
    if not schema_sql_path.exists():
        return [f"Schema file not found: {schema_sql_path}"]

    # 1. Create an in-memory DB and load the ideal schema
    ideal_conn = sqlite3.connect(":memory:")
    try:
        with schema_sql_path.open("r", encoding="utf-8") as f:
            ideal_conn.executescript(f.read())
        ideal_schema = get_schema_dict(ideal_conn)
    finally:
        ideal_conn.close()

    # 2. Open the actual DB and extract its schema
    actual_conn = sqlite3.connect(str(db_path))
    try:
        actual_schema = get_schema_dict(actual_conn)
    finally:
        actual_conn.close()

    errors: List[str] = []

    # Check for missing or mismatched tables
    for table, ideal_sql in ideal_schema.items():
        if table not in actual_schema:
            errors.append(f"Missing table: {table}")
        else:
            clean_ideal = " ".join(ideal_sql.split()) if ideal_sql else ""
            clean_actual = (
                " ".join(actual_schema[table].split()) if actual_schema[table] else ""
            )
            if clean_ideal != clean_actual:
                errors.append(f"Schema mismatch in table '{table}'.")

    # Check for unexpected extra tables
    for table in actual_schema:
        if table not in ideal_schema:
            errors.append(f"Unexpected extra table found: {table}")

    return errors
