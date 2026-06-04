"""Statistics computation and immediate visualization using Polars."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import polars as pl
from tabulate import tabulate

from src.dataloader import IndividualProjectContext


@dataclass
class MetricConfig:
    """Configuration mapping for running and plotting specific metrics dynamically."""

    key: str
    filename: str
    title_suffix: str
    ylabel: str
    xlabel: str
    transform_fn: Callable[[IndividualProjectContext], pl.DataFrame]
    is_status_plot: bool = False
    x_col: Optional[str] = None
    y_col: Optional[str] = None


# =====================================================================
# Data Transformation Registry
# =====================================================================


def _get_revisions_df(ctx: IndividualProjectContext) -> pl.DataFrame:
    return (
        ctx.proposal_revisions.with_columns(year=pl.col("created_at").dt.year())
        .group_by("year")
        .agg(pl.col("revision_index").count().alias("revision_count"))
        .sort("year")
    )


def _get_comments_df(ctx: IndividualProjectContext) -> pl.DataFrame:
    return (
        ctx.comments.with_columns(year=pl.col("created_at").dt.year())
        .group_by("year")
        .agg(pl.len().alias("comment_count"))
        .sort("year")
    )


def _get_authors_df(ctx: IndividualProjectContext) -> pl.DataFrame:
    return (
        ctx.proposal_revisions.join(
            ctx.proposal_revision_authors,
            on=["project_id", "proposal_id", "revision_index"],
            how="inner",
        )
        .with_columns(year=pl.col("created_at").dt.year())
        .group_by("year")
        .agg(pl.col("author_id").n_unique().alias("author_count"))
        .sort("year")
    )


def _get_statuses_df(ctx: IndividualProjectContext) -> pl.DataFrame:
    return (
        ctx.proposal_statuses.with_columns(year=pl.col("created_at").dt.year())
        .group_by(["year", "normalised_status"])
        .agg(pl.col("proposal_id").count().alias("status_count"))
        .sort(["year", "normalised_status"])
    )


def _get_domains_df(ctx: IndividualProjectContext) -> pl.DataFrame:
    return (
        ctx.person_identifiers.group_by("domain")
        .agg(pl.col("person_id").n_unique().alias("count"))
        .sort("count", descending=True)
    )


def _get_identifiers_df(ctx: IndividualProjectContext) -> pl.DataFrame:
    return (
        ctx.person_identifiers.group_by("identifier_type")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )


def _get_organisations_df(ctx: IndividualProjectContext) -> pl.DataFrame:
    return (
        ctx.organisations.join(ctx.affiliations, on="organisation_id", how="left")
        .group_by("organisation_name")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .limit(20)
    )


# Centralized Configuration Registry replacing the switch statement setup
METRIC_REGISTRY = [
    MetricConfig(
        "revisions",
        "revisions_per_year.png",
        "Revisions",
        "Count",
        "Year",
        _get_revisions_df,
        x_col="year",
        y_col="revision_count",
    ),
    MetricConfig(
        "comments",
        "comments_per_year.png",
        "Comments",
        "Count",
        "Year",
        _get_comments_df,
        x_col="year",
        y_col="comment_count",
    ),
    MetricConfig(
        "authors",
        "authors_proposing_per_year.png",
        "Authors",
        "Count",
        "Year",
        _get_authors_df,
        x_col="year",
        y_col="author_count",
    ),
    MetricConfig(
        "statuses",
        "proposal_status_per_year.png",
        "Status Timeline",
        "Number of Proposals",
        "Year",
        _get_statuses_df,
        is_status_plot=True,
    ),
    MetricConfig(
        "domains",
        "person_identifiers_domain_counts.png",
        "Domains",
        "Count",
        "Domain",
        _get_domains_df,
        x_col="domain",
        y_col="count",
    ),
    MetricConfig(
        "identifiers",
        "person_identifiers_type_counts.png",
        "ID Types",
        "Count",
        "Type",
        _get_identifiers_df,
        x_col="identifier_type",
        y_col="count",
    ),
    MetricConfig(
        "organisations",
        "organisations_name_counts.png",
        "Orgs",
        "Count",
        "Organisation",
        _get_organisations_df,
        x_col="organisation_name",
        y_col="count",
    ),
]

# =====================================================================
# UI Engine & Plotting Mechanics
# =====================================================================


def _render_axis(ax: plt.Axes, df: pl.DataFrame, cfg: MetricConfig, title: str) -> None:
    """Determines chart styling and dynamically renders the data onto an explicit plot axis."""
    if df.is_empty() or (
        cfg.is_status_plot and df.drop_nulls(["year", "normalised_status"]).is_empty()
    ):
        ax.text(
            0.5,
            0.5,
            f"No Data Available for\n{title}",
            fontsize=10,
            weight="bold",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="darkred",
        )
    elif cfg.is_status_plot:
        clean_df = df.drop_nulls(["year", "normalised_status"])
        for status_val in clean_df.select("normalised_status").unique().to_series():
            data = clean_df.filter(pl.col("normalised_status") == status_val).sort(
                "year"
            )
            ax.plot(
                data.select("year").to_series().to_list(),
                data.select("status_count").to_series().to_list(),
                marker="o",
                label=status_val,
                linewidth=2,
            )
        if ax.get_lines():
            ax.legend(fontsize=8)
    else:
        x_data = [
            str(x) if x is not None else "Unknown"
            for x in df.select(pl.col(cfg.x_col)).to_series().to_list()
        ]
        y_data = df.select(pl.col(cfg.y_col)).to_series().to_list()
        ax.bar(x_data, y_data, alpha=0.7, edgecolor="black")
        ax.tick_params(axis="x", rotation=45)

    ax.set_xlabel(cfg.xlabel, fontsize=10)
    ax.set_ylabel(cfg.ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.grid(True, alpha=0.3, axis="y" if not cfg.is_status_plot else "both")


def _build_combined_grid(
    contexts: list[IndividualProjectContext], cfg: MetricConfig, output_dir: Path
) -> None:
    """Builds a balanced square grid containing subplots of this metric for all projects."""
    n = len(contexts)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    flat_axes = axes.flatten() if n > 1 else [axes]

    for idx, ctx in enumerate(contexts):
        _render_axis(flat_axes[idx], cfg.transform_fn(ctx), cfg, ctx.project_name)

    for empty_idx in range(idx + 1, len(flat_axes)):
        flat_axes[empty_idx].set_axis_off()

    plt.suptitle(
        f"Combined Grid: {cfg.key.title()} Analysis Across All Projects",
        fontsize=16,
        weight="bold",
    )
    plt.tight_layout()
    plt.savefig(str(output_dir / f"combined_{cfg.key}_grid.png"), dpi=150)
    plt.close()


# =====================================================================
# Unified Entrypoint
# =====================================================================


def show_basic_statistics(
    projects: list[IndividualProjectContext], output_dir: Path
) -> None:
    """Processes datasets individually per project, storing tracking outputs and square grid overviews."""

    # Loop 1: Render isolated visual graph subdirectories per project safely
    for ctx in projects:
        proj_folder = (
            output_dir
            / f"{ctx.project_id}_{ctx.project_name.lower().replace(' ', '_')}"
        )
        proj_folder.mkdir(parents=True, exist_ok=True)

        for cfg in METRIC_REGISTRY:
            fig, ax = plt.subplots(figsize=(10, 5))
            _render_axis(
                ax,
                cfg.transform_fn(ctx),
                cfg,
                f"{ctx.project_name}: {cfg.title_suffix}",
            )
            plt.tight_layout()
            plt.savefig(str(proj_folder / cfg.filename), dpi=150)
            plt.close()

    # Loop 2: Compute and save Combined Square Grid Overviews directly to output root folder
    output_dir.mkdir(parents=True, exist_ok=True)
    for cfg in METRIC_REGISTRY:
        _build_combined_grid(projects, cfg, output_dir)


def generate_table_counts(
    projects: list[IndividualProjectContext], output_path: Path
) -> None:
    """Generates tracking metrics documents (Markdown and LaTeX) detailing shape allocations per project."""
    if not projects:
        return

    table_fields = [
        ("Proposal", "proposals"),
        ("Proposal revisions", "proposal_revisions"),
        ("Proposal revision authors", "proposal_revision_authors"),
        ("Proposal statuses", "proposal_statuses"),
        ("Comment", "comments"),
        ("Person identifiers", "person_identifiers"),
        ("Organisations", "organisations"),
        ("Affiliations", "affiliations"),
    ]

    # --- 1. SWAP LAYOUT: Build Transposed Data ---
    md_headers = ["Project"] + [label for label, _ in table_fields]
    latex_headers = ["Project"] + [label for label, _ in table_fields]

    table_data = []
    for ctx in projects:
        row = [ctx.project_name]
        for label, attr in table_fields:
            df = getattr(ctx, attr)
            count = (
                df.select("proposal_id").n_unique() if label == "Proposal" else len(df)
            )
            row.append(count)
        table_data.append(row)

    # --- 2. Ensure output directory exists ---
    output_path.mkdir(parents=True, exist_ok=True)

    # --- 3. Generate and save Markdown Table ---
    markdown_table = tabulate(table_data, headers=md_headers, tablefmt="github")
    markdown_output = f"# Table Item Counts by Project\n\n{markdown_table}\n"

    with open(output_path / "table_counts.md", "w") as f:
        f.write(markdown_output)

    # --- 4. Generate and save LaTeX Table matching template ---
    num_data_cols = len(table_fields)

    latex_lines = [
        r"\begin{table*}[tb]",
        r"    \centering",
        r"    \caption{Overview of dataset counts across different programming languages and technologies.}",
        r"    \label{tab:dataset_counts}",
        # Define a custom centered wrapping column layout using tabularx
        r"    \newcolumntype{Y}{>{\centering\arraybackslash}X}",
        f"    \\begin{{tabularx}}{{\\textwidth}}{{l*{{{num_data_cols}}}{{Y}}}}",
        r"    \hline",
    ]

    # Escape underscores for LaTeX safety
    escaped_headers = [h.replace("_", r"\_") for h in latex_headers]
    latex_lines.append("    " + " & ".join(escaped_headers) + r" \\")
    latex_lines.append(r"    \hline")

    # Populate numerical data rows
    for row in table_data:
        str_row = [str(item).replace("_", r"\_") for item in row]
        latex_lines.append("    " + " & ".join(str_row) + r" \\")

    latex_lines.append(r"    \hline")
    latex_lines.append(r"    \end{tabularx}")
    latex_lines.append(r"\end{table*}")

    latex_table = "\n".join(latex_lines)

    with open(output_path / "table_counts.tex", "w") as f:
        f.write(latex_table)
