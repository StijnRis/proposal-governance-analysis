"""Main orchestration script for proposal governance analysis."""

import datetime
from pathlib import Path

from dotenv import load_dotenv

from dataloader import load_all_projects
from enhance_data.add_companies import enrich_project_contexts_with_companies
from enhance_data.merge_companies import merge_duplicate_companies_in_contexts
from enhance_data.merge_people import merge_duplicate_people
from governance_calc import get_governance_statistics
from governance_plots import show_governance_in_plots
from governance_stats import save_governance_statistics
from health_check import diagnose_all_projects, save_combined_report
from statistics2 import generate_table_counts, show_basic_statistics


def main() -> None:
    """Load data, compute statistics, and generate visualizations."""
    print("Starting proposal governance analysis...")
    load_dotenv()
    BASE_FONT_SIZE = 21

    # Setup core operational path markers
    data_dir = Path("data/proposals")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path("output")
    output_dir = base_output_dir / timestamp

    # Clean output dir of 2 or more runs ago
    dirs = sorted(
        base_output_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for old_dir in dirs[2:]:
        if old_dir.is_dir():
            for item in old_dir.glob("*"):
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    for subitem in item.glob("*"):
                        if subitem.is_file():
                            subitem.unlink()
                    item.rmdir()
            old_dir.rmdir()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all SQLite databases
    print(f"Looking for SQLite databases in: {data_dir.resolve()}")
    db_files = []
    for ext in ["*.sqlite3", "*.db", "*.sqlite"]:
        db_files.extend(data_dir.glob(ext))

    if not db_files:
        raise FileNotFoundError(
            f"No SQLite databases found in target folder: {data_dir}"
        )

    projects = load_all_projects(db_files, max_proposals=None)
    projects = merge_duplicate_people(projects)
    projects = enrich_project_contexts_with_companies(projects)
    projects = merge_duplicate_companies_in_contexts(projects, output_dir)

    reports = diagnose_all_projects(projects)
    save_combined_report(reports, output_dir)

    generate_table_counts(projects, reports, output_dir)
    show_basic_statistics(projects, output_dir)

    project_governance_stats = get_governance_statistics(projects)
    show_governance_in_plots(project_governance_stats, output_dir, BASE_FONT_SIZE)
    save_governance_statistics(project_governance_stats, output_dir)


if __name__ == "__main__":
    main()
