"""Main orchestration script for proposal governance analysis."""

import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from statistics import calculate_all_statistics

import pandas as pd

from visualizations import (
    generate_all_visualizations,
    plot_all_projects_governance_lines,
)


def generate_tables_with_pandas(
    stats: Dict[str, Any], output_dir: str = "output"
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    project_names = list(stats.keys())
    features = [
        "Proposals",
        "Revisions",
        "Authors",
        "Comments",
        "Stages",
        "Success Rate (%)",
        "Avg Comments per Proposal",
        "Avg Unique Authors per Proposal",
    ]
    data = {"Feature": features}

    for proj in project_names:
        counts = stats[proj].get("basic_counts", {})
        success = stats[proj].get("success_rate", 0.0)
        avg_comments = stats[proj].get("avg_comments_per_proposal", 0.0)
        avg_authors = stats[proj].get("avg_unique_authors_per_proposal", 0.0)

        data[proj] = [
            counts.get("num_proposals", 0),
            counts.get("num_revisions", 0),
            counts.get("num_authors", 0),
            counts.get("num_comments", 0),
            counts.get("num_stages", 0),
            f"{success:.2f}",
            f"{avg_comments:.2f}",
            f"{avg_authors:.2f}",
        ]

    df = pd.DataFrame(data)

    # Save markdown
    md_path = os.path.join(output_dir, "project_statistics.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Project Statistics Summary\n\n")
        f.write(df.to_markdown(index=False))

    # Save LaTeX
    tex_path = os.path.join(output_dir, "project_statistics.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, escape=False))


def main() -> None:
    """Main pipeline for statistics and visualization generation."""
    load_dotenv()
    # Collect database paths from multiple sources:
    # - DATABASE_LOCATION env var (single file)
    # - all files in ./data with common sqlite extensions
    # - EXTRA_DATABASE_LOCATIONS env var (comma/pathsep/semicolon/newline separated)
    candidate_paths = []

    # Files under data/ with common DB extensions
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if os.path.isdir(data_dir):
        for fname in os.listdir(data_dir):
            fpath = os.path.join(data_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith(
                (".db", ".sqlite", ".sqlite3", ".db3")
            ):
                candidate_paths.append(fpath)

    # EXTRA_DATABASE_LOCATIONS may contain multiple paths separated by OS pathsep, commas, semicolons or newlines
    extra = os.getenv("EXTRA_DATABASE_LOCATIONS", "")
    if extra:
        parts = []
        for sep in [os.pathsep, ",", ";", "\n"]:
            if sep in extra:
                parts = [
                    p.strip()
                    for p in extra.replace(os.pathsep, ",").replace(";", ",").split(",")
                ]
                break
        if not parts:
            parts = [extra.strip()]
        for p in parts:
            if p:
                candidate_paths.append(p)

    # Deduplicate and keep only existing files
    seen = set()
    db_paths = []
    for p in candidate_paths:
        p_norm = os.path.abspath(p)
        if p_norm in seen:
            continue
        seen.add(p_norm)
        if os.path.exists(p_norm) and os.path.isfile(p_norm):
            db_paths.append(p_norm)

    if not db_paths:
        print(
            "ERROR: No database files found. Set DATABASE_LOCATION, place DBs in ./data, or set EXTRA_DATABASE_LOCATIONS."
        )
        sys.exit(1)

    # Process each database separately, writing outputs to project folders under ./output
    # Collect per-database stats and defer creating combined aggregate until after
    # all databases are processed to avoid incremental "all-project" calculations.
    collected_stats: list[Dict[str, Any]] = []
    for db in db_paths:
        out_dir = os.path.join("output")
        os.makedirs(out_dir, exist_ok=True)

        print(f"Using database: {db}")
        try:
            print("Calculating statistics...")
            stats = calculate_all_statistics(db)
            # Collect stats for later aggregation
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
                proj_out = os.path.join(out_dir, f"{project_id}_{slug}")
                os.makedirs(proj_out, exist_ok=True)

                print(
                    f"Generating visualizations for {project_name} (project_id={project_id})..."
                )
                # Generate visualizations and tables for this single project into its folder
                generate_all_visualizations(
                    {project_name: project_stats}, output_dir=proj_out
                )
                generate_tables_with_pandas(
                    {project_name: project_stats}, output_dir=proj_out
                )

            print(f"Completed processing {db}. Outputs written to {out_dir}")
        except Exception as exc:
            print(f"ERROR processing {db}: {exc}")
            # continue with next database
            continue

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

    try:
        plot_all_projects_governance_lines(aggregate_stats, output_dir="output")
    except Exception:
        pass

    print("All tasks completed. Check the output directory for results.")


if __name__ == "__main__":
    main()
