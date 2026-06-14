from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import polars as pl

from statistics2 import IndividualProjectContext

# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------


@dataclass
class NullRateRow:
    table: str
    column: str
    total_rows: int
    null_count: int
    null_pct: float  # 0-100


@dataclass
class RelationCoverageRow:
    relation: str  # human-readable label, e.g. "proposal → revision"
    source_entity: str  # e.g. "proposal_id"
    source_count: int  # distinct source entities
    covered_count: int  # sources with ≥ 1 relation
    coverage_pct: float  # covered / total * 100
    mean_relations: float  # avg relation count incl. zeros


@dataclass
class ProjectHealthReport:
    project_name: str
    null_rates: List[NullRateRow] = field(default_factory=list)
    relation_coverage: List[RelationCoverageRow] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Null-rate helpers
# ---------------------------------------------------------------------------


def _null_rates_for_table(table_name: str, df: pl.DataFrame) -> List[NullRateRow]:
    rows = []
    n = len(df)
    for col in df.columns:
        null_count = df[col].null_count()
        rows.append(
            NullRateRow(
                table=table_name,
                column=col,
                total_rows=n,
                null_count=null_count,
                null_pct=round(null_count / n * 100, 2) if n > 0 else 0.0,
            )
        )
    return rows


def _compute_null_rates(ctx: IndividualProjectContext) -> List[NullRateRow]:
    tables = {
        "proposals": ctx.proposals,
        "proposal_revisions": ctx.proposal_revisions,
        "proposal_revision_authors": ctx.proposal_revision_authors,
        "proposal_statuses": ctx.proposal_statuses,
        "comments": ctx.comments,
        "persons": ctx.persons,
        "person_identifiers": ctx.person_identifiers,
        "organisations": ctx.organisations,
        "affiliations": ctx.affiliations,
        "related_proposals": ctx.related_proposals,
    }
    rows: List[NullRateRow] = []
    for name, df in tables.items():
        rows.extend(_null_rates_for_table(name, df))
    return rows


# ---------------------------------------------------------------------------
# Relation-coverage helper
# ---------------------------------------------------------------------------


def _relation_coverage(
    label: str,
    source_col: str,
    source_df: pl.DataFrame,
    target_df: pl.DataFrame,
    join_cols: List[str],
) -> RelationCoverageRow:
    """
    Computes coverage and mean-relations for one FK relationship.
    """
    # Distinct source entities
    all_sources = source_df.select(join_cols).unique()
    source_count = len(all_sources)

    if source_count == 0:
        return RelationCoverageRow(label, source_col, 0, 0, 0.0, 0.0)

    # Count relations per source entity
    relation_counts = target_df.group_by(join_cols).agg(pl.len().alias("n_relations"))

    merged = all_sources.join(relation_counts, on=join_cols, how="left").with_columns(
        pl.col("n_relations").fill_null(0)
    )

    covered = int((merged["n_relations"] > 0).sum())
    mean_rel = float(merged["n_relations"].mean() or 0.0)

    return RelationCoverageRow(
        relation=label,
        source_entity=source_col,
        source_count=source_count,
        covered_count=covered,
        coverage_pct=round(covered / source_count * 100, 2),
        mean_relations=round(mean_rel, 3),
    )


def _compute_relation_coverage(
    ctx: IndividualProjectContext,
) -> List[RelationCoverageRow]:
    rows: List[RelationCoverageRow] = []

    # proposal → revisions
    rows.append(
        _relation_coverage(
            label="proposal → revision",
            source_col="proposal_id",
            source_df=ctx.proposals.select(["proposal_id"]),
            target_df=ctx.proposal_revisions.select(["proposal_id", "revision_index"]),
            join_cols=["proposal_id"],
        )
    )

    # proposal_revision → authors
    if not ctx.proposal_revisions.is_empty():
        rev_keys = ctx.proposal_revisions.select(
            ["proposal_id", "revision_index"]
        ).unique()
        rows.append(
            _relation_coverage(
                label="proposal_revision → author",
                source_col="(proposal_id, revision_index)",
                source_df=rev_keys,
                target_df=ctx.proposal_revision_authors.select(
                    ["proposal_id", "revision_index", "author_id"]
                ),
                join_cols=["proposal_id", "revision_index"],
            )
        )

    # proposal → status entries
    rows.append(
        _relation_coverage(
            label="proposal → status",
            source_col="proposal_id",
            source_df=ctx.proposals.select(["proposal_id"]),
            target_df=ctx.proposal_statuses.select(["proposal_id", "status_index"]),
            join_cols=["proposal_id"],
        )
    )

    # proposal → comments
    rows.append(
        _relation_coverage(
            label="proposal → comment",
            source_col="proposal_id",
            source_df=ctx.proposals.select(["proposal_id"]),
            target_df=ctx.comments.filter(pl.col("proposal_id").is_not_null()).select(
                ["proposal_id", "comment_id"]
            ),
            join_cols=["proposal_id"],
        )
    )

    # person → affiliation
    if not ctx.persons.is_empty():
        rows.append(
            _relation_coverage(
                label="person → affiliation (org)",
                source_col="person_id",
                source_df=ctx.persons.select(["person_id"]),
                target_df=ctx.affiliations.select(["person_id", "organisation_id"]),
                join_cols=["person_id"],
            )
        )

    # person → identifiers
    if not ctx.persons.is_empty():
        rows.append(
            _relation_coverage(
                label="person → identifier",
                source_col="person_id",
                source_df=ctx.persons.select(["person_id"]),
                target_df=ctx.person_identifiers.select(["person_id", "domain"]),
                join_cols=["person_id"],
            )
        )

    # person → authored revisions
    if not ctx.persons.is_empty():
        rows.append(
            _relation_coverage(
                label="person → authored revision",
                source_col="person_id",
                source_df=ctx.persons.select(["person_id"]),
                target_df=ctx.proposal_revision_authors.rename(
                    {"author_id": "person_id"}
                ).select(["person_id", "proposal_id"]),
                join_cols=["person_id"],
            )
        )

    # person → comments written
    if not ctx.persons.is_empty():
        rows.append(
            _relation_coverage(
                label="person → comment written",
                source_col="person_id",
                source_df=ctx.persons.select(["person_id"]),
                target_df=ctx.comments.filter(pl.col("author_id").is_not_null())
                .rename({"author_id": "person_id"})
                .select(["person_id", "comment_id"]),
                join_cols=["person_id"],
            )
        )

    # proposal → related proposals
    rows.append(
        _relation_coverage(
            label="proposal → related proposal",
            source_col="proposal_id",
            source_df=ctx.proposals.select(["proposal_id"]),
            target_df=ctx.related_proposals.select(
                ["proposal_id", "related_proposal_id"]
            ),
            join_cols=["proposal_id"],
        )
    )

    return rows


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def diagnose_project(ctx: IndividualProjectContext) -> ProjectHealthReport:
    """Run all health checks for a single project context."""
    return ProjectHealthReport(
        project_name=ctx.project_name,
        null_rates=_compute_null_rates(ctx),
        relation_coverage=_compute_relation_coverage(ctx),
    )


def diagnose_all_projects(
    contexts: List[IndividualProjectContext],
) -> List[ProjectHealthReport]:
    """Run diagnostics for every project context."""
    return [diagnose_project(ctx) for ctx in contexts]


# ---------------------------------------------------------------------------
# Consolidated Markdown report helper
# ---------------------------------------------------------------------------


def save_combined_report(reports: List[ProjectHealthReport], output_dir: Path) -> None:
    """Save a single combined Markdown summary across all evaluated projects."""
    if not reports:
        return

    output_path = output_dir / "combined_projects_health_report.md"
    project_names = [r.project_name for r in reports]

    # --- Pre-process Matrix Maps ---
    # Map from (table, col) -> {project_name: null_pct}
    null_matrix: Dict[Tuple[str, str], Dict[str, float]] = {}
    # Map from relation_label -> {project_name: (coverage_pct, mean_relations)}
    relation_matrix: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for report in reports:
        for nr in report.null_rates:
            key = (nr.table, nr.column)
            if key not in null_matrix:
                null_matrix[key] = {}
            null_matrix[key][report.project_name] = nr.null_pct

        for rc in report.relation_coverage:
            if rc.relation not in relation_matrix:
                relation_matrix[rc.relation] = {}
            relation_matrix[rc.relation][report.project_name] = (
                rc.coverage_pct,
                rc.mean_relations,
            )

    # Sort the row keys gracefully
    sorted_null_keys = sorted(null_matrix.keys(), key=lambda x: (x[0], x[1]))
    sorted_relation_keys = sorted(relation_matrix.keys())

    with open(output_path, "w", encoding="utf-8") as f:
        # Title
        f.write("# Master Data Health Report\n\n")
        f.write(f"**Evaluated Projects:** {', '.join(project_names)}\n\n")

        # --- [1] Consolidated Null Rates ---
        f.write("## [1] Attribute Null Rates\n\n")

        # Headers
        null_header = (
            "| Table | Column | "
            + " | ".join(f"{name} Null %" for name in project_names)
            + " |\n"
        )
        null_divider = (
            "| :--- | :--- | " + " | ".join("---:" for _ in project_names) + " |\n"
        )
        f.write(null_header)
        f.write(null_divider)

        # Content rows
        for table, col in sorted_null_keys:
            row_str = f"| {table} | {col} |"
            for name in project_names:
                pct = null_matrix[(table, col)].get(name, None)
                if pct is not None:
                    flag = " ⚠️" if pct > 10.0 else ""
                    row_str += f" {pct:.1f}%{flag} |"
                else:
                    row_str += " N/A |"
            f.write(row_str + "\n")

        f.write("\n")

        # --- [2] Consolidated Relation Coverage ---
        f.write("## [2] Relation Coverage Matrix\n\n")

        # Headers (Project Sub-headers built dynamically)
        rel_header = (
            "| Relation | "
            + " | ".join(f"{name} Cov % | {name} Mean #" for name in project_names)
            + " |\n"
        )
        rel_divider = (
            "| :--- | " + " | ".join("---: | ---:" for _ in project_names) + " |\n"
        )
        f.write(rel_header)
        f.write(rel_divider)

        # Content rows
        for relation in sorted_relation_keys:
            row_str = f"| {relation} |"
            for name in project_names:
                metrics = relation_matrix[relation].get(name, None)
                if metrics is not None:
                    cov_pct, mean_rel = metrics
                    flag = " ⚠️" if cov_pct < 80.0 else ""
                    row_str += f" {cov_pct:.1f}%{flag} | {mean_rel:.2f} |"
                else:
                    row_str += " N/A | N/A |"
            f.write(row_str + "\n")
