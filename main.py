"""Main orchestration script for proposal governance analysis."""

from pathlib import Path

from dotenv import load_dotenv

from src.governance import show_governance_statistics
from src.dataloader import load_all_projects
from src.statistics import generate_table_counts_markdown, show_basic_statistics


def main() -> None:
    """Load data, compute statistics, and generate visualizations."""
    load_dotenv()

    # Setup core operational path markers
    data_dir = Path("data")
    output_dir = Path("output")

    # Clean output dir
    for item in output_dir.glob("*"):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            for subitem in item.glob("*"):
                if subitem.is_file():
                    subitem.unlink()
            item.rmdir()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all SQLite databases
    db_files = []
    for ext in ["*.sqlite3", "*.db", "*.sqlite"]:
        db_files.extend(data_dir.glob(ext))

    if not db_files:
        raise FileNotFoundError(
            f"No SQLite databases found in target folder: {data_dir}"
        )

    projects = load_all_projects(db_files)

    generate_table_counts_markdown(projects, output_dir)
    show_basic_statistics(projects, output_dir)
    show_governance_statistics(projects, output_dir)
    

    # # Trigger comparative plotting run
    # if all_metrics_results:
    #     print("\n📊 Generating consolidated metrics plots over time...")
    #     plot_governance_dimensions(all_metrics_results, output_dir)
    # else:
    #     print("\n⚠ No valid project data parsed. Skipping plot orchestration.")

    # # Generate table counts markdown
    # print("\n📋 Generating table counts markdown...")
    # generate_table_counts_markdown(project_data, output_dir)

    # print(f"\n✓ Analysis complete. Output saved to {output_dir}")


if __name__ == "__main__":
    main()
