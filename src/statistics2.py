import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import polars as pl
import tabulate

from dataloader import IndividualProjectContext
from health_check import ProjectHealthReport


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
    status_col: Optional[str] = None  # Added to dynamically identify the status column
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


def _get_raw_statuses_df(ctx: IndividualProjectContext) -> pl.DataFrame:
    """Groups and counts metrics by year and raw_status."""
    return (
        ctx.proposal_statuses.with_columns(year=pl.col("created_at").dt.year())
        .group_by(["year", "raw_status"])
        .agg(pl.col("proposal_id").count().alias("status_count"))
        .sort(["year", "raw_status"])
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
        "revisions_per_year.svg",
        "Revisions",
        "Count",
        "Year",
        _get_revisions_df,
        x_col="year",
        y_col="revision_count",
    ),
    MetricConfig(
        "comments",
        "comments_per_year.svg",
        "Comments",
        "Count",
        "Year",
        _get_comments_df,
        x_col="year",
        y_col="comment_count",
    ),
    MetricConfig(
        "authors",
        "authors_proposing_per_year.svg",
        "Authors",
        "Count",
        "Year",
        _get_authors_df,
        x_col="year",
        y_col="author_count",
    ),
    MetricConfig(
        "statuses",
        "proposal_status_per_year.svg",
        "Status Timeline (Normalized)",
        "Number of Proposals",
        "Year",
        _get_statuses_df,
        is_status_plot=True,
        status_col="normalised_status",
    ),
    MetricConfig(
        "raw_statuses",
        "proposal_raw_status_per_year.svg",
        "Status Timeline (Raw)",
        "Number of Proposals",
        "Year",
        _get_raw_statuses_df,
        is_status_plot=True,
        status_col="raw_status",
    ),
    MetricConfig(
        "domains",
        "person_identifiers_domain_counts.svg",
        "Domains",
        "Count",
        "Domain",
        _get_domains_df,
        x_col="domain",
        y_col="count",
    ),
    MetricConfig(
        "identifiers",
        "person_identifiers_type_counts.svg",
        "ID Types",
        "Count",
        "Type",
        _get_identifiers_df,
        x_col="identifier_type",
        y_col="count",
    ),
    MetricConfig(
        "organisations",
        "organisations_name_counts.svg",
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
        cfg.is_status_plot and df.drop_nulls(["year", cfg.status_col]).is_empty()
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
        # Dynamically uses cfg.status_col instead of hardcoded 'normalised_status'
        clean_df = df.drop_nulls(["year", cfg.status_col])
        for status_val in clean_df.select(cfg.status_col).unique().to_series():
            data = clean_df.filter(pl.col(cfg.status_col) == status_val).sort("year")
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
        ax.set_xticks(range(len(x_data)))
        ax.set_xticklabels(x_data, rotation=45, ha="right", rotation_mode="anchor")

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
        f"Combined Grid: {cfg.key.replace('_', ' ').title()} Analysis Across All Projects",
        fontsize=16,
        weight="bold",
    )
    plt.tight_layout()
    plt.savefig(str(output_dir / f"combined_{cfg.key}_grid.svg"))
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
            plt.savefig(str(proj_folder / cfg.filename))
            plt.close()

    # Loop 2: Compute and save Combined Square Grid Overviews directly to output root folder
    output_dir.mkdir(parents=True, exist_ok=True)
    for cfg in METRIC_REGISTRY:
        _build_combined_grid(projects, cfg, output_dir)


def generate_table_counts(
    projects: list["IndividualProjectContext"],
    reports: List["ProjectHealthReport"],
    output_path: Path,
) -> None:
    """Generates tracking metrics documents (Markdown and LaTeX) detailing shape allocations per project.

    The layout is transposed: Properties/Tables occupy the rows (Y-axis), while Projects occupy the columns (X-axis).
    """
    if not projects:
        return

    # Logical hierarchy mapping: (Markdown Display Name, LaTeX Display Name, Context Attribute Name)
    table_fields = [
        ("Proposals", "proposals"),
        ("Related proposals", "related_proposals"),
        ("Revisions", "proposal_revisions"),
        ("Statuses", "proposal_statuses"),
        ("Comments", "comments"),
        ("Authors", "proposal_revision_authors"),
        ("People", "persons"),
        ("Person identifiers", "person_identifiers"),
        ("Organizations", "organisations"),
        ("Affiliations", "affiliations"),
        ("% revisions with authors", "pct_authored"),
        ("% people with affiliations", "pct_affiliated"),
    ]

    report_map = {r.project_name: r for r in reports}
    project_names = [ctx.project_name for ctx in projects]

    # --- 1. BUILD TRANSPOSED MATRIX (Rows = Properties, Columns = Projects) ---
    md_headers = ["Property"] + project_names

    # We will build two distinct data matrices to cleanly separate markdown text from raw LaTeX macros
    md_table_data = []
    latex_table_data = []

    for label, attr in table_fields:
        md_row = [label]
        latex_row = [label.replace("%", "\\%")]

        for ctx in projects:
            if attr.startswith("pct_"):
                report = report_map.get(ctx.project_name)
                coverage_val = None
                if report:
                    for rc in report.relation_coverage:
                        if (
                            rc.relation == "person → affiliation (org)"
                            and attr == "pct_affiliated"
                        ):
                            coverage_val = rc.coverage_pct
                            break
                        elif (
                            rc.relation == "proposal_revision → author"
                            and attr == "pct_authored"
                        ):
                            coverage_val = rc.coverage_pct
                            break

                if coverage_val is None:
                    raise ValueError(
                        f"Missing coverage metric for {ctx.project_name} in report."
                    )

                val_str_md = f"{coverage_val:.1f}%"
                val_str_ltx = f"\\num{{{coverage_val:.1f}}}\\%"
            else:
                df = getattr(ctx, attr)
                # Keep original behavior: exact length check
                count = len(df)
                val_str_md = str(count)
                val_str_ltx = f"\\num{{{count}}}"

            md_row.append(val_str_md)
            latex_row.append(val_str_ltx)

        md_table_data.append(md_row)
        latex_table_data.append(latex_row)

    # --- 2. Ensure output directory exists ---
    output_path.mkdir(parents=True, exist_ok=True)

    # --- 3. Generate and save Markdown Table ---
    col_alignments_md = ["left"] + ["right"] * len(projects)
    markdown_table = tabulate.tabulate(
        md_table_data, headers=md_headers, tablefmt="github", colalign=col_alignments_md
    )
    markdown_output = f"# Table Item Counts Across Projects\n\n{markdown_table}\n"

    with open(output_path / "table_counts.md", "w", encoding="utf-8") as f:
        f.write(markdown_output)

    # --- 4. Generate and save LaTeX Table matching template ---
    num_data_cols = len(projects)

    latex_lines = [
        r"\begin{table*}[tb]",
        r"    \centering",
        r"    \caption{Dataset counts across ten SEP systems.}",
        r"    \label{tab:dataset_counts}",
        r"    \small",
        r"    \setlength{\tabcolsep}{3pt}",
        r"    ",
        r"    % Y is our standard right-aligned data column type",
        r"    \newcolumntype{Y}{>{\raggedleft\arraybackslash}X}",
        r"    ",
        r"    ",
        f"    \\begin{{tabularx}}{{\\textwidth}}{{l*{{{num_data_cols}}}{{Y}}}}",
        r"    \toprule",
    ]

    # Generate Headers with rotation macro applied to the project names on the X-axis
    latex_headers = ["Property"]
    for name in project_names:
        # Sanitize project names containing underscores for LaTeX
        clean_name = name.replace("_", r"\_")
        latex_headers.append(f"{clean_name}")

    latex_lines.append("    " + " & ".join(latex_headers) + r" \\")
    latex_lines.append(r"    \midrule")

    # Append rows from the pre-formatted latex matrix
    for row in latex_table_data:
        latex_lines.append("    " + " & ".join(row) + r" \\")

    latex_lines.append(r"    \bottomrule")
    latex_lines.append(r"    \end{tabularx}")
    latex_lines.append(r"\end{table*}")

    latex_table = "\n".join(latex_lines)

    with open(output_path / "table_counts.tex", "w", encoding="utf-8") as f:
        f.write(latex_table)
