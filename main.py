"""Main orchestration script for proposal governance analysis."""
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from statistics import calculate_all_statistics
from visualizations import generate_all_visualizations
import pandas as pd


def generate_tables_with_pandas(stats, output_dir="output"):
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


def main():
    """Main pipeline for statistics and visualization generation."""
    load_dotenv()
    db_path = os.getenv("DATABASE_LOCATION")

    if not db_path:
        print("ERROR: DATABASE_LOCATION not set in .env")
        sys.exit(1)

    if not os.path.exists(db_path):
        print(f"ERROR: Database file not found at {db_path}")
        sys.exit(1)

    print(f"Using database: {db_path}")
    
    print("Calculating statistics...")
    stats = calculate_all_statistics(db_path)

    print("Generating visualizations...")
    generate_all_visualizations(stats, output_dir="output")

    print("Generating tables...")
    generate_tables_with_pandas(stats, output_dir="output")

    print("All tasks completed. Check the output directory for results.")


if __name__ == "__main__":
    main()