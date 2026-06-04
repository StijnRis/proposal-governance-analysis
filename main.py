"""Main orchestration script for proposal governance analysis."""

import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.governance import show_governance_statistics
from src.dataloader import load_all_projects
from src.statistics import show_basic_statistics, generate_table_counts


def main() -> None:
    """Load data, compute statistics, and generate visualizations."""
    print("Starting proposal governance analysis...")
    load_dotenv()

    # Setup core operational path markers
    data_dir = Path("data")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path("output")
    output_dir = base_output_dir / timestamp

    # Clean output dir of 2 or more runs ago
    dirs = sorted(base_output_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
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
    db_files = []
    for ext in ["*.sqlite3", "*.db", "*.sqlite"]:
        db_files.extend(data_dir.glob(ext))
    
    if not db_files:
        raise FileNotFoundError(
            f"No SQLite databases found in target folder: {data_dir}"
        )

    projects = load_all_projects(db_files, max_proposals=None)

    generate_table_counts(projects, output_dir)
    show_basic_statistics(projects, output_dir)
    show_governance_statistics(projects, output_dir)

if __name__ == "__main__":
    main()
