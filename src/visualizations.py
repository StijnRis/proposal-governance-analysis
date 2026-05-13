"""Visualization generation for proposal governance statistics."""
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import pandas as pd


def ensure_output_dir(output_dir: str = "output") -> None:
    """Ensure output directory exists."""
    os.makedirs(output_dir, exist_ok=True)


def plot_revisions_over_time(stats: Dict, project_name: str, output_dir: str = "output") -> None:
    """Plot proposal revisions per year as a bar chart."""
    df = stats[project_name].get("revisions_over_time")

    if df is None or df.empty:
        return

    # Ensure datetime
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["year"] = df["created_at"].dt.year

    counts = df.groupby("year").size().sort_index()
    if counts.empty:
        return

    plt.figure(figsize=(12, 6))
    plt.bar(counts.index.astype(str), counts.values, edgecolor="black", alpha=0.7)
    plt.xlabel("Year")
    plt.ylabel("Number of Revisions")
    plt.title(f"{project_name}: Revisions per Year")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{project_name.lower().replace(' ', '_')}_revisions_timeline.png"), dpi=300)
    plt.close()


def plot_proposals_over_time(stats: Dict, project_name: str, output_dir: str = "output") -> None:
    """Plot number of proposals created per year as a bar chart."""
    df = stats[project_name].get("proposals_over_time")
    if df is None or df.empty:
        return

    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["year"] = df["created_at"].dt.year

    counts = df.groupby("year").size().sort_index()
    if counts.empty:
        return

    plt.figure(figsize=(12, 6))
    plt.bar(counts.index.astype(str), counts.values, edgecolor="black", alpha=0.7)
    plt.xlabel("Year")
    plt.ylabel("Number of Proposals")
    plt.title(f"{project_name}: Proposals per Year")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{project_name.lower().replace(' ', '_')}_proposals_timeline.png"), dpi=300)
    plt.close()


def plot_author_tenure_distribution(stats: Dict, project_name: str, output_dir: str = "output") -> None:
    """Plot histogram of author tenure duration."""
    tenure_data = stats[project_name]["author_tenure_distribution"]
    durations = tenure_data["duration_days"]

    if not durations:
        return

    plt.figure(figsize=(12, 6))
    plt.hist(durations, bins=30, edgecolor="black", alpha=0.7)
    plt.xlabel("Author Tenure (Days)")
    plt.ylabel("Number of Authors")
    plt.title(f"{project_name}: Author Tenure Distribution")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{project_name.lower().replace(' ', '_')}_author_tenure.png"), dpi=300)
    plt.close()


def plot_author_activity_distribution(stats: Dict, project_name: str, output_dir: str = "output") -> None:
    """Plot author activity distribution (top N authors by revision count)."""
    activity_df = stats[project_name]["author_activity_distribution"]

    if activity_df.empty:
        return

    # Show top 20 authors or fewer if not enough data
    top_n = min(20, len(activity_df))
    top_authors = activity_df.head(top_n)

    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(top_authors)), top_authors["revision_count"].values, edgecolor="black", alpha=0.7)
    plt.xlabel("Author Rank")
    plt.ylabel("Number of Revisions")
    plt.title(f"{project_name}: Top {top_n} Most Active Authors")
    plt.xticks(range(len(top_authors)), [f"#{i+1}" for i in range(len(top_authors))])
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{project_name.lower().replace(' ', '_')}_author_activity.png"), dpi=300
    )
    plt.close()


def plot_comments_distribution(stats: Dict, project_name: str, output_dir: str = "output") -> None:
    """Plot distribution of comments per proposal."""
    comments_df = stats[project_name]["comments_per_proposal"]

    if comments_df.empty:
        return

    plt.figure(figsize=(12, 6))
    plt.hist(comments_df["comment_count"], bins=30, edgecolor="black", alpha=0.7)
    plt.xlabel("Number of Comments per Proposal")
    plt.ylabel("Number of Proposals")
    plt.title(f"{project_name}: Comment Distribution")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{project_name.lower().replace(' ', '_')}_comments_distribution.png"), dpi=300)
    plt.close()


def generate_all_visualizations(stats: Dict, output_dir: str = "output") -> None:
    """Generate all visualizations for all projects."""
    ensure_output_dir(output_dir)

    for project_name in stats.keys():
        print(f"Generating visualizations for {project_name}...")
        plot_revisions_over_time(stats, project_name, output_dir)
        plot_proposals_over_time(stats, project_name, output_dir)

    print(f"✓ All visualizations saved to {output_dir}/")
