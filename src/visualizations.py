"""Visualization generation for proposal governance statistics.

Adds:
- time-series plot for the five governance dimensions per project
- radar plot for the last year of each project
"""

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# shared time parsing utility
from src.timeutils import to_naive_series as _to_naive_series

# default font sizes for clearer labels
_TITLE_FS = 30
_LABEL_FS = 24
_TICK_FS = 20
_LEGEND_FS = 20


def ensure_output_dir(output_dir: Path) -> None:
    """Ensure output directory exists."""
    outp = Path(output_dir)
    outp.mkdir(parents=True, exist_ok=True)


def plot_revisions_over_time(
    stats: Dict[str, Any], project_name: str, output_dir: Path
) -> None:
    """Plot proposal revisions per year as a bar chart."""
    df = stats[project_name].get("revisions_over_time")

    if df is None or df.empty:
        return

    # Ensure datetime
    df = df.copy()
    df["created_at"] = _to_naive_series(df["created_at"])
    df["year"] = df["created_at"].dt.year

    counts = df.groupby("year").size().sort_index()
    if counts.empty:
        return

    plt.figure(figsize=(12, 6))
    plt.bar(counts.index.astype(str), counts.values, edgecolor="black", alpha=0.7)
    plt.xlabel("Year", fontsize=_LABEL_FS)
    plt.ylabel("Number of Revisions", fontsize=_LABEL_FS)
    plt.title(f"{project_name}: Revisions per Year", fontsize=_TITLE_FS)
    plt.xticks(fontsize=_TICK_FS)
    plt.yticks(fontsize=_TICK_FS)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    outp = Path(output_dir)
    fname = f"{project_name.lower().replace(' ', '_')}_revisions_timeline.png"
    plt.savefig(str(outp / fname), dpi=300)
    plt.close()


def plot_authors_proposing_per_year(
    stats: Dict[str, Any], project_name: str, output_dir: Path
) -> None:
    """Plot number of unique authors who proposed revisions per year."""
    data = stats[project_name].get("authors_proposing_per_year", {})
    if not data:
        return

    years = sorted(data.keys())
    counts = [data[y] for y in years]

    plt.figure(figsize=(12, 6))
    plt.bar([str(y) for y in years], counts, edgecolor="black", alpha=0.7)
    plt.xlabel("Year", fontsize=_LABEL_FS)
    plt.ylabel("Number of Authors", fontsize=_LABEL_FS)
    plt.title(
        f"{project_name}: Authors Proposing Revisions per Year", fontsize=_TITLE_FS
    )
    plt.xticks(fontsize=_TICK_FS)
    plt.yticks(fontsize=_TICK_FS)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    outp = Path(output_dir)
    fname = f"{project_name.lower().replace(' ', '_')}_authors_proposing_per_year.png"
    plt.savefig(str(outp / fname), dpi=300)
    plt.close()


def plot_authors_commenting_per_year(
    stats: Dict[str, Any], project_name: str, output_dir: Path
) -> None:
    """Plot number of unique authors who left comments per year."""
    data = stats[project_name].get("authors_commenting_per_year", {})
    if not data:
        return

    years = sorted(data.keys())
    counts = [data[y] for y in years]

    plt.figure(figsize=(12, 6))
    plt.bar([str(y) for y in years], counts, edgecolor="black", alpha=0.7)
    plt.xlabel("Year", fontsize=_LABEL_FS)
    plt.ylabel("Number of Authors", fontsize=_LABEL_FS)
    plt.title(f"{project_name}: Authors Commenting per Year", fontsize=_TITLE_FS)
    plt.xticks(fontsize=_TICK_FS)
    plt.yticks(fontsize=_TICK_FS)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    outp = Path(output_dir)
    fname = f"{project_name.lower().replace(' ', '_')}_authors_commenting_per_year.png"
    plt.savefig(str(outp / fname), dpi=300)
    plt.close()


def plot_authors_proposing_and_commenting_distinct_per_year(
    stats: Dict[str, Any], project_name: str, output_dir: Path
) -> None:
    """Combine distinct authors proposing and commenting per year into one plot.

    Uses the pre-computed per-year unique-author counts from `statistics.py`.
    Rotates x-axis labels by 45 degrees for readability.
    """
    prop = stats[project_name].get("authors_proposing_per_year", {}) or {}
    comm = stats[project_name].get("authors_commenting_per_year", {}) or {}

    # union of years across both series
    years = sorted(set(list(prop.keys()) + list(comm.keys())))
    if not years:
        return

    prop_counts = [int(prop.get(y, 0)) for y in years]
    comm_counts = [int(comm.get(y, 0)) for y in years]

    x = np.arange(len(years))
    width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, prop_counts, width=width, label="Proposing (distinct)", edgecolor="black", alpha=0.8)
    plt.bar(x + width / 2, comm_counts, width=width, label="Commenting (distinct)", edgecolor="black", alpha=0.8)

    plt.xlabel("Year", fontsize=_LABEL_FS)
    plt.ylabel("Number of Distinct Authors", fontsize=_LABEL_FS)
    plt.title(f"{project_name}: Distinct Authors Proposing vs Commenting per Year", fontsize=_TITLE_FS)
    plt.xticks(x, [str(y) for y in years], fontsize=_TICK_FS, rotation=45)
    plt.yticks(fontsize=_TICK_FS)
    plt.legend(fontsize=_LEGEND_FS)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    outp = Path(output_dir)
    fname = f"{project_name.lower().replace(' ', '_')}_authors_proposing_commenting_per_year.png"
    plt.savefig(str(outp / fname), dpi=300)
    plt.close()


def plot_proposals_over_time(
    stats: Dict[str, Any], project_name: str, output_dir: Path
) -> None:
    """Plot number of proposals created per year as a bar chart."""
    df = stats[project_name].get("proposals_over_time")
    if df is None or df.empty:
        return

    df = df.copy()
    df["created_at"] = _to_naive_series(df["created_at"])
    df["year"] = df["created_at"].dt.year

    counts = df.groupby("year").size().sort_index()
    if counts.empty:
        return

    plt.figure(figsize=(12, 6))
    plt.bar(counts.index.astype(str), counts.values, edgecolor="black", alpha=0.7)
    plt.xlabel("Year", fontsize=_LABEL_FS)
    plt.ylabel("Number of Proposals", fontsize=_LABEL_FS)
    plt.title(f"{project_name}: Proposals per Year", fontsize=_TITLE_FS)
    plt.xticks(fontsize=_TICK_FS)
    plt.yticks(fontsize=_TICK_FS)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    outp = Path(output_dir)
    fname = f"{project_name.lower().replace(' ', '_')}_proposals_timeline.png"
    plt.savefig(str(outp / fname), dpi=300)
    plt.close()


def plot_author_tenure_distribution(
    stats: Dict[str, Any], project_name: str, output_dir: Path
) -> None:
    """Plot histogram of author tenure duration."""
    tenure_data = stats[project_name].get("author_tenure_distribution", {})
    durations = tenure_data.get("duration_days", [])

    if not durations:
        return

    plt.figure(figsize=(12, 6))
    plt.hist(durations, bins=30, edgecolor="black", alpha=0.7)
    plt.xlabel("Author Tenure (Days)", fontsize=_LABEL_FS)
    plt.ylabel("Number of Authors", fontsize=_LABEL_FS)
    plt.title(f"{project_name}: Author Tenure Distribution", fontsize=_TITLE_FS)
    plt.xticks(fontsize=_TICK_FS)
    plt.yticks(fontsize=_TICK_FS)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    outp = Path(output_dir)
    fname = f"{project_name.lower().replace(' ', '_')}_author_tenure.png"
    plt.savefig(str(outp / fname), dpi=300)
    plt.close()


def plot_author_activity_distribution(
    stats: Dict[str, Any], project_name: str, output_dir: Path
) -> None:
    """Plot author activity distribution (top N authors by revision count)."""
    activity_df = stats[project_name].get(
        "author_activity_distribution", pd.DataFrame()
    )

    if activity_df is None or activity_df.empty:
        return

    # Show top 20 authors or fewer if not enough data
    top_n = min(20, len(activity_df))
    top_authors = activity_df.head(top_n)

    plt.figure(figsize=(14, 6))
    bars = plt.bar(
        range(len(top_authors)),
        top_authors["revision_count"].values,
        edgecolor="black",
        alpha=0.7,
    )
    plt.xlabel("Author Rank", fontsize=_LABEL_FS)
    plt.ylabel("Number of Revisions", fontsize=_LABEL_FS)
    plt.title(f"{project_name}: Top {top_n} Most Active Authors", fontsize=_TITLE_FS)
    plt.xticks(
        range(len(top_authors)),
        [f"#{i + 1}" for i in range(len(top_authors))],
        fontsize=_TICK_FS,
    )
    plt.yticks(fontsize=_TICK_FS)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    outp = Path(output_dir)
    fname = f"{project_name.lower().replace(' ', '_')}_author_activity.png"
    plt.savefig(str(outp / fname), dpi=300)
    plt.close()


def plot_comments_distribution(
    stats: Dict[str, Any], project_name: str, output_dir: Path
) -> None:
    """Plot distribution of comments per proposal."""
    comments_df = stats[project_name].get("comments_per_proposal", pd.DataFrame())

    if comments_df is None or comments_df.empty:
        return

    plt.figure(figsize=(12, 6))
    plt.hist(comments_df["comment_count"], bins=30, edgecolor="black", alpha=0.7)
    plt.xlabel("Number of Comments per Proposal", fontsize=_LABEL_FS)
    plt.ylabel("Number of Proposals", fontsize=_LABEL_FS)
    plt.title(f"{project_name}: Comment Distribution", fontsize=_TITLE_FS)
    plt.xticks(fontsize=_TICK_FS)
    plt.yticks(fontsize=_TICK_FS)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    outp = Path(output_dir)
    fname = f"{project_name.lower().replace(' ', '_')}_comments_distribution.png"
    plt.savefig(str(outp / fname), dpi=300)
    plt.close()


def plot_governance_timeseries(
    stats: Dict[str, Any], project_name: str, output_dir: Path
) -> None:
    """Plot the five governance dimensions as time series on one chart."""
    gm = stats[project_name].get("governance_metrics")
    if not gm:
        return
    years = gm.get("years", [])
    if not years:
        return

    years_sorted = sorted(years)
    # remove the most recent year from analysis
    if len(years_sorted) > 1:
        years_sorted = years_sorted[:-1]

    if not years_sorted:
        return

    ind = [gm["independence_hhi"].get(y, 0.0) for y in years_sorted]
    pl = [gm["pluralism_gini"].get(y, 0.0) for y in years_sorted]
    rep = [gm["representation_gini"].get(y, 0.0) for y in years_sorted]
    cen = [gm["centralization"].get(y, 0.0) for y in years_sorted]
    new = [gm["newcomer_success"].get(y, 0.0) for y in years_sorted]

    plt.figure(figsize=(12, 6))
    # short legend labels
    plt.plot(years_sorted, ind, marker="o", label="Independence")
    plt.plot(years_sorted, pl, marker="o", label="Pluralism")
    plt.plot(years_sorted, rep, marker="o", label="Representation")
    plt.plot(years_sorted, cen, marker="o", label="Centralization")
    plt.plot(years_sorted, new, marker="o", label="NewcomerSuccess")

    plt.xlabel("Year", fontsize=_LABEL_FS)
    plt.ylabel("Score", fontsize=_LABEL_FS)
    plt.ylim(0, 1)
    plt.title(f"{project_name} governance metrics over time", fontsize=_TITLE_FS)
    plt.xticks(fontsize=_TICK_FS)
    plt.yticks(fontsize=_TICK_FS)
    plt.legend(fontsize=_LEGEND_FS)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    outp = Path(output_dir)
    fname = f"{project_name.lower().replace(' ', '_')}_governance_timeseries.png"
    plt.savefig(str(outp / fname), dpi=300)
    plt.close()


def plot_governance_radar(
    stats: Dict[str, Any], project_name: str, output_dir: Path
) -> None:
    """Create a radar plot for the last year available for the project."""
    gm = stats[project_name].get("governance_metrics")
    if not gm:
        return
    years = gm.get("years", [])
    if not years:
        return
    # Prefer 2025 for the radar if available, otherwise use last available year
    years_sorted = sorted(years)
    last_year = 2025 if 2025 in years_sorted else years_sorted[-1]
    values = [
        gm["independence_hhi"].get(last_year, 0.0),
        gm["pluralism_gini"].get(last_year, 0.0),
        gm["representation_gini"].get(last_year, 0.0),
        gm["centralization"].get(last_year, 0.0),
        gm["newcomer_success"].get(last_year, 0.0),
    ]
    labels = [
        "Independence",
        "Pluralism",
        "Representation",
        "Centralization",
        "NewcomerSuccess",
    ]

    # prepare radar
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="tab:blue", linewidth=2)
    ax.fill(angles, values, color="tab:blue", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=_TICK_FS)
    ax.set_ylim(0, 1)
    ax.set_title(f"{project_name} governance radar ({last_year})", fontsize=_TITLE_FS)
    plt.tight_layout()
    # use descriptive filename
    fname = f"{project_name.lower().replace(' ', '_')}_governance_radar_{last_year}.png"
    outp = Path(output_dir)
    plt.savefig(str(outp / fname), dpi=300)
    plt.close()


def plot_proposal_stage_counts_per_year(
    stats: Dict[str, Any], project_name: str, output_dir: Path
) -> None:
    """Create a single stacked bar chart: one bar per year, stacks per stage.

    Each bar's height is the total number of proposals accounted for that year;
    segments show counts per `normalised_status`.
    """
    data = stats[project_name].get("proposal_stage_counts_per_year", {})
    if not data:
        return

    years = sorted(data.keys())
    # collect all stages across years for consistent ordering
    all_stages = sorted({s for yc in data.values() for s in yc.keys()})
    if not years or not all_stages:
        return

    # build matrix of counts: rows=stages, cols=years
    counts_matrix = []
    for stage in all_stages:
        counts_matrix.append([data.get(y, {}).get(stage, 0) for y in years])

    # colors
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(all_stages))]

    # stacked bar plot
    plt.figure(figsize=(12, 6))
    bottoms = [0] * len(years)
    for idx, stage in enumerate(all_stages):
        vals = counts_matrix[idx]
        plt.bar(
            [str(y) for y in years],
            vals,
            bottom=bottoms,
            color=colors[idx],
            label=stage,
        )
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    plt.xlabel("Year", fontsize=_LABEL_FS)
    plt.ylabel("Number of Proposals", fontsize=_LABEL_FS)
    plt.title(
        f"{project_name}: Proposal Stage Distribution per Year", fontsize=_TITLE_FS
    )
    plt.xticks(fontsize=_TICK_FS)
    plt.yticks(fontsize=_TICK_FS)
    plt.legend(
        title="Stage",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=_LEGEND_FS,
        title_fontsize=_LEGEND_FS,
    )
    plt.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    fname = (
        f"{project_name.lower().replace(' ', '_')}_proposal_stage_counts_stacked.png"
    )
    outp = Path(output_dir)
    plt.savefig(str(outp / fname), dpi=300)
    plt.close()


def save_governance_timeseries_md(
    stats: Dict[str, Any], project_name: str, output_dir: Path
) -> None:
    """Save governance timeseries for a project to a markdown table."""
    gm = stats[project_name].get("governance_metrics")
    if not gm:
        return
    years = sorted(gm.get("years", []))
    if not years:
        return
    # remove the most recent year from analysis
    if len(years) > 1:
        years = years[:-1]
    outp = Path(output_dir)
    md_path = (
        outp / f"{project_name.lower().replace(' ', '_')}_governance_timeseries.md"
    )
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {project_name} Governance Timeseries\n\n")
        f.write(
            "| Year | Independence | Pluralism | Representation | Centralization | NewcomerSuccess |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for y in years:
            ind = gm["independence_hhi"].get(y, 0.0)
            pl = gm["pluralism_gini"].get(y, 0.0)
            rep = gm["representation_gini"].get(y, 0.0)
            cen = gm["centralization"].get(y, 0.0)
            new = gm["newcomer_success"].get(y, 0.0)
            f.write(
                f"| {y} | {ind:.3f} | {pl:.3f} | {rep:.3f} | {cen:.3f} | {new:.3f} |\n"
            )


def generate_all_visualizations(stats: Dict[str, Any], output_dir: Path) -> None:
    """Generate all visualizations for all projects."""
    ensure_output_dir(output_dir)
    outp = Path(output_dir)

    for project_name in stats.keys():
        print(f"Generating visualizations for {project_name}...")
        plot_revisions_over_time(stats, project_name, output_dir)
        plot_authors_proposing_and_commenting_distinct_per_year(
            stats, project_name, output_dir
        )
        plot_proposals_over_time(stats, project_name, output_dir)
        plot_proposal_stage_counts_per_year(stats, project_name, output_dir)
        plot_author_tenure_distribution(stats, project_name, output_dir)
        plot_author_activity_distribution(stats, project_name, output_dir)
        plot_comments_distribution(stats, project_name, output_dir)
        plot_governance_timeseries(stats, project_name, output_dir)
        # save markdown timeseries table
        save_governance_timeseries_md(stats, project_name, output_dir)
        plot_governance_radar(stats, project_name, output_dir)

    print(f"✓ All visualizations saved to {outp.name}/")


def plot_all_projects_governance_lines(stats: Dict[str, Any], output_dir: Path) -> None:
    """Create one line plot per governance metric with all projects over years.

    Each plot's y-axis is fixed to [0, 1]. The most recent year is removed per-project.
    """
    metrics = [
        ("independence_hhi", "Independence"),
        ("pluralism_gini", "Pluralism"),
        ("representation_gini", "Representation"),
        ("centralization", "Centralization"),
        ("newcomer_success", "Newcomer success"),
    ]

    for key, label in metrics:
        plt.figure(figsize=(12, 6))
        any_plotted = False
        for project_name, pdata in stats.items():
            gm = pdata.get("governance_metrics")
            if not gm:
                continue
            years = sorted(gm.get("years", []))
            if len(years) > 1:
                years = years[:-1]
            if not years:
                continue
            ys = [gm.get(key, {}).get(y, 0.0) for y in years]
            # Use the project's declared name as legend label when available
            label_name = (
                pdata.get("project_name") if isinstance(pdata, dict) else project_name
            )
            plt.plot(years, ys, marker="o", label=str(label_name))
            any_plotted = True

        if not any_plotted:
            plt.close()
            continue

        plt.xlabel("Year", fontsize=_LABEL_FS)
        plt.ylabel("Score", fontsize=_LABEL_FS)
        plt.ylim(0, 1)
        plt.title(f"{label} over time", fontsize=_TITLE_FS)
        plt.xticks(fontsize=_TICK_FS)
        plt.yticks(fontsize=_TICK_FS)
        plt.legend(loc="best", fontsize=_LEGEND_FS)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f"all_projects_{key}_timeseries.png"
        plt.savefig(str(output_dir / fname), dpi=300)
        plt.close()
