import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import stats

from governance_calc import GovernanceProjectStats, MetricInterval


def _resolve_attr_name(dim_str: str) -> str:
    """Maps display strings or existing dict keys to the new dataclass attribute names."""
    mapping = {
        "Independence": "independence",
        "Pluralism": "pluralism",
        "Representation": "representation",
        "Decentralized Decision-Making": "decentralization",
        "Autonomous Participation": "autonomous_participation",
    }
    # Return string if it matches precisely or lower-fallback to guarantee match
    return mapping.get(dim_str, dim_str.lower().replace(" ", "_").replace("-", "_"))


def _get_metric_data(
    p_stat: GovernanceProjectStats, dim: str, mode: str = "pooled", year: int = None
) -> Any:
    """Helper method to clean up object attribute extraction via dot-notation wrapper."""
    attr_name = _resolve_attr_name(dim)

    if mode == "pooled":
        profile = getattr(p_stat.pooled_metrics, attr_name, None)
        return profile if profile is not None else MetricInterval(0.0, 0.0, 0.0)

    elif mode == "timeline":
        timeline_profile = getattr(p_stat.metrics, attr_name, None)
        if timeline_profile and year in timeline_profile.windows:
            return timeline_profile.windows[year]
        return MetricInterval(0.0, 0.0, 0.0)

    return MetricInterval(0.0, 0.0, 0.0)


def calculate_age_vs_autonomy_correlation(
    project_data: List[GovernanceProjectStats],
) -> Dict[str, float]:

    ages = []
    autonomy_scores = []

    for project in project_data:
        score = project.pooled_metrics.autonomous_participation.val
        age = -project.release_year

        ages.append(age)
        autonomy_scores.append(score)

    # Fallback checking if there are insufficient data points to run correlation
    if len(ages) < 3:
        raise ValueError("Insufficient overlapping data points found")

    # Compute the statistical metrics
    pearson_r, pearson_p = stats.pearsonr(ages, autonomy_scores)
    spearman_rho, spearman_p = stats.spearmanr(ages, autonomy_scores)

    return {
        "sample_size": float(len(ages)),
        "pearson_r": float(pearson_r),
        "pearson_p_value": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p_value": float(spearman_p),
    }


def save_governance_statistics(
    project_data: List[GovernanceProjectStats], output_dir: Path
) -> None:
    """Compute and display governance statistics for the given project data."""

    correlation_results = calculate_age_vs_autonomy_correlation(project_data)

    # Save the correlation results to a JSON file
    output_file = output_dir / "age_vs_autonomy_statistics.json"
    with open(output_file, "w") as f:
        json.dump(correlation_results, f, indent=4)

    print(f"Governance statistics saved to {output_file}")


@dataclass(frozen=True)
class CorrelationMethodResult:
    corr_df: pd.DataFrame
    p_values_df: pd.DataFrame
    ci_bounds: Dict[Tuple[int, int], Tuple[float, float]]


CorrelationResults = Dict[Literal["pearson", "spearman"], CorrelationMethodResult]


def calculate_dimensions_correlation(
    projects_stats: List["GovernanceProjectStats"],
    dimensions: List[str],
) -> CorrelationResults:
    """Computes correlation matrices, p-values, and analytical 95% confidence intervals

    expressed as relative errors, optimized for small sample sizes (N=10).
    """
    n_samples = len(projects_stats)
    if len(dimensions) < 2 or n_samples < 4:
        return {}

    # Extract data into DataFrame
    df = pd.DataFrame(
        {
            dim: [_get_metric_data(p, dim, mode="pooled").val for p in projects_stats]
            for dim in dimensions
        }
    ).fillna(0.0)

    results: CorrelationResults = {}
    methods: List[Literal["pearson", "spearman"]] = ["pearson", "spearman"]

    for method in methods:
        # Initialize empty matrices for correlations and p-values
        num_dims = len(dimensions)
        corr_matrix = np.eye(num_dims)
        p_matrix = np.zeros((num_dims, num_dims))
        ci_bounds: Dict[Tuple[int, int], Tuple[float, float]] = {}

        # Calculate pairwise metrics
        for i, dim_row in enumerate(dimensions):
            for j, dim_col in enumerate(dimensions):
                if i == j:
                    ci_bounds[(i, j)] = (0.0, 0.0)
                    continue

                x = df[dim_row].values
                y = df[dim_col].values

                # Calculate correlation coefficient and p-value
                try:
                    if method == "pearson":
                        r_val, p_val = stats.pearsonr(x, y)
                    else:  # spearman
                        # scipy automatically handles small-sample approximations
                        res = stats.spearmanr(x, y)
                        r_val, p_val = res.statistic, res.pvalue
                except Exception:
                    r_val, p_val = 0.0, 1.0

                corr_matrix[i, j] = r_val
                p_matrix[i, j] = p_val

                # Calculate Confidence Intervals for lower triangle to match original logic
                if i > j:
                    r_clamped = max(min(r_val, 0.9999), -0.9999)
                    try:
                        # 1. Fisher Z-Transformation
                        z_stat = np.arctanh(r_clamped)

                        # 2. Standard Error (with Spearman correction factor)
                        se_modifier = 1.06 if method == "spearman" else 1.0
                        se = se_modifier / np.sqrt(n_samples - 3)

                        # 3. Critical Value using Student's t-distribution
                        t_crit = stats.t.ppf(1 - 0.05 / 2, df=n_samples - 2)

                        # Calculate absolute bounds in Z-space
                        z_low = z_stat - (t_crit * se)
                        z_high = z_stat + (t_crit * se)

                        # 4. Transform back to original r-space
                        ci_low = float(np.tanh(z_low))
                        ci_high = float(np.tanh(z_high))

                        # 5. Convert to relative error distances for your plotting script
                        lower_err = max(0.0, r_val - ci_low)
                        upper_err = max(0.0, ci_high - r_val)

                        ci_bounds[(i, j)] = (lower_err, upper_err)
                    except Exception:
                        ci_bounds[(i, j)] = (0.0, 0.0)
                else:
                    # Maintain structural integrity for upper triangle mapping if needed
                    if i < j:
                        ci_bounds[(i, j)] = (0.0, 0.0)

        # Convert matrices back to organized DataFrames
        corr_df = pd.DataFrame(corr_matrix, index=dimensions, columns=dimensions)
        p_values_df = pd.DataFrame(p_matrix, index=dimensions, columns=dimensions)

        results[method] = CorrelationMethodResult(
            corr_df=corr_df, p_values_df=p_values_df, ci_bounds=ci_bounds
        )

    return results
