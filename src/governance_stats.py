import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
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
    output_file = output_dir / "governance_statistics.json"
    with open(output_file, "w") as f:
        json.dump(correlation_results, f, indent=4)

    print(f"Governance statistics saved to {output_file}")


def calculate_dimensions_correlation(
    projects_stats: List["GovernanceProjectStats"],
    dimensions: List[str],
    n_resamples: int = 2000,
) -> Dict[str, dict]:
    """Computes correlation matrices and their bootstrapped 95% confidence intervals.

    Returns:
        A dictionary mapping method ('pearson', 'spearman') to its results:
        {
            "method_name": {
                "corr_df": pd.DataFrame,
                "ci_bounds": Dict[Tuple[int, int], Tuple[float, float]]  # (i, j) -> (lower_err, upper_err)
            }
        }
    """
    if len(dimensions) < 2 or not projects_stats:
        return {}

    # Structure raw data into a Pandas DataFrame
    df = pd.DataFrame(
        {
            dim: [_get_metric_data(p, dim, mode="pooled").val for p in projects_stats]
            for dim in dimensions
        }
    ).fillna(0.0)

    results = {}
    methods = ["pearson", "spearman"]

    for method in methods:
        corr_df = df.corr(method=method).fillna(0.0)
        corr_matrix = corr_df.values
        ci_bounds = {}

        # Helper metric function for the bootstrap process
        def _corr_metric(x, y):
            if method == "pearson":
                return stats.pearsonr(x, y)[0]
            else:
                return stats.spearmanr(x, y)[0]

        # Calculate CIs for the lower triangle
        for i, dim_row in enumerate(dimensions):
            for j, dim_col in enumerate(dimensions):
                if i > j:  # Lower triangle only
                    x_data = df[dim_row].values
                    y_data = df[dim_col].values

                    try:
                        res = stats.bootstrap(
                            (x_data, y_data),
                            _corr_metric,
                            vectorized=False,
                            paired=True,
                            n_resamples=n_resamples,
                            method="BCa",
                            random_state=42,
                        )
                        ci_low = res.confidence_interval.low
                        ci_high = res.confidence_interval.high

                        # Convert absolute CI bounds to relative error values around the core estimate
                        r_val = corr_matrix[i, j]
                        lower_err = max(0.0, r_val - ci_low)
                        upper_err = max(0.0, ci_high - r_val)
                        ci_bounds[(i, j)] = (lower_err, upper_err)

                    except Exception:
                        ci_bounds[(i, j)] = (0.0, 0.0)
                else:
                    # Identity diagonal or upper triangle (masked visually anyway)
                    ci_bounds[(i, j)] = (0.0, 0.0)

        results[method] = {"corr_df": corr_df, "ci_bounds": ci_bounds}

    return results
