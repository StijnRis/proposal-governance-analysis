"""Main orchestration script for proposal governance analysis."""

import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from src.dbschema import attach_and_create_union_views, verify_schema
from src.statistics import calculate_all_statistics
from src.visualizations import (
    generate_all_visualizations,
    plot_all_projects_governance_lines,
)


def generate_tables_with_pandas(
    stats: Dict[str, Any],
    output_dir: Path,
    db_path: Optional[Path] = None,
    db_conn: Optional[sqlite3.Connection] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    project_names = list(stats.keys())
    features = [
        "Success Rate (%)",
        "Avg Comments per Proposal",
        "Avg Unique Authors per Proposal",
    ]
    data = {"Feature": features}

    for proj in project_names:
        success = stats[proj].get("success_rate", 0.0)
        avg_comments = stats[proj].get("avg_comments_per_proposal", 0.0)
        avg_authors = stats[proj].get("avg_unique_authors_per_proposal", 0.0)

        data[proj] = [
            f"{success:.2f}",
            f"{avg_comments:.2f}",
            f"{avg_authors:.2f}",
        ]

    df = pd.DataFrame(data)

    # Save markdown
    md_path = output_dir / "project_statistics.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Project Statistics Summary\n\n")
        f.write(df.to_markdown(index=False))

        conn_to_close = None
        try:
            if db_conn is not None:
                cur = db_conn.cursor()
            elif db_path is not None:
                conn_to_close = sqlite3.connect(str(db_path))
                cur = conn_to_close.cursor()
            else:
                cur = None

            if cur is not None:
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
                tables = [r[0] for r in cur.fetchall()]
                f.write("\n\n## Table row counts\n\n")
                f.write("| Table | Rows |\n")
                f.write("|---|---:|\n")
                for t in tables:
                    try:
                        cur.execute(f'SELECT COUNT(*) FROM "{t}"')
                        cnt = cur.fetchone()[0]
                    except Exception:
                        cnt = "-"
                    f.write(f"| {t} | {cnt} |\n")
        finally:
            if conn_to_close is not None:
                conn_to_close.close()

    # Save LaTeX
    tex_path = output_dir / "project_statistics.tex"
    with tex_path.open("w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, escape=False))


def main() -> None:
    """Main pipeline for statistics and visualization generation."""
    load_dotenv()

    output_dir = Path("output")

    # Clean output directory before starting
    shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover database files to process
    db_paths = []
    data_dir = Path(os.path.dirname(__file__)) / "data"
    if data_dir.is_dir():
        for f in data_dir.iterdir():
            if f.is_file() and f.suffix.lower() in (
                ".db",
                ".sqlite",
                ".sqlite3",
                ".db3",
            ):
                db_paths.append(f)

    if not db_paths:
        raise ValueError(
            "ERROR: No database files found. Set DATABASE_LOCATION, place DBs in ./data, or set EXTRA_DATABASE_LOCATIONS."
        )

    # Process each database separately, writing outputs to project folders under ./output
    # Collect per-database stats and defer creating combined aggregate until after
    # all databases are processed to avoid incremental "all-project" calculations.
    collected_stats: list[Dict[str, Any]] = []
    # Verify all DBs first and collect valid ones
    schema_file = Path("database_schema.sql")
    valid_dbs: List[Path] = []
    for db in db_paths:
        mismatches = verify_schema(schema_file, db)
        if mismatches:
            print(f"❌ Schema validation failed for {db.name}:")
            for err in mismatches:
                print(f"  - {err}")
            print("Skipping this database for analysis.")
            continue
        valid_dbs.append(db)

    if not valid_dbs:
        raise ValueError("No valid databases found after schema verification.")

    # Open a primary connection to the first valid DB and attach the others.
    primary_db = valid_dbs[0]
    conn = sqlite3.connect(str(primary_db))
    # attach_and_create_union_views will attach the remaining DBs and create TEMP VIEWs
    attach_and_create_union_views(conn, schema_file, valid_dbs[1:])

    print("Calculating statistics across all attached databases...")
    stats = calculate_all_statistics(conn)
    collected_stats.append(stats)

    # For each project create a project-specific folder based on project_id
    for project_name, project_stats in stats.items():
        project_id = project_stats.get("project_id")
        # slugify project name for folder
        raw_name = str(project_name or "unknown")
        slug = (
            "".join(c for c in raw_name if c.isalnum() or c.isspace())
            .replace(" ", "_")
            .lower()
        )
        proj_out = output_dir / f"{project_id}_{slug}"
        proj_out.mkdir(parents=True, exist_ok=True)

        print(
            f"Generating visualizations for {project_name} (project_id={project_id})..."
        )
        # Generate visualizations and tables for this single project into its folder
        generate_all_visualizations({project_name: project_stats}, output_dir=proj_out)
        generate_tables_with_pandas(
            {project_name: project_stats}, output_dir=proj_out, db_conn=conn
        )

    # close the combined connection
    conn.close()

    # After all DBs processed, create combined plots across all discovered projects
    # Build a single aggregate mapping of all projects from collected stats
    aggregate_stats: Dict[str, Any] = {}
    for stats in collected_stats:
        for project_name, project_stats in stats.items():
            pid = project_stats.get("project_id")
            agg_key = f"{pid}_{project_name}"
            if agg_key in aggregate_stats:
                suffix = 1
                while f"{agg_key}_{suffix}" in aggregate_stats:
                    suffix += 1
                agg_key = f"{agg_key}_{suffix}"
            aggregate_stats[agg_key] = project_stats

    plot_all_projects_governance_lines(aggregate_stats, output_dir=output_dir)

    print("All tasks completed. Check the output directory for results.")


if __name__ == "__main__":
    main()
