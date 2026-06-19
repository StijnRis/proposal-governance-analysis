> **Notice:** This repository and its contents should be used for **research purposes only**.


# Proposal Governance Analysis

## Overview

Research codebase for analysing how governance metrics evolve across multiple open-source projects by mining and processing enhancement proposals.

This project has been made as part of the research project 2026 at TU Delft (https://github.com/TU-Delft-CSE/Research-Project).

### Key Points

* **Language:** Python 3.13
* **Environment:** `uv` managing a local virtualenv (see `pyproject.toml`)
* **Purpose:** Extract, transform, and analyse proposal data to produce governance statistics and reports

## Quick Start

### 1. Create and Activate a Virtual Environment (Recommended)

```powershell
uv venv create
uv venv activate
```

### 2. Install Dependencies

The project uses `uv` tasks via `pyproject.toml`:

```powershell
uv install
```

### 3. Run the Main Pipeline

```powershell
uv run main.py

```

## Testing

Run unit tests for core logic:

```powershell
uv run pytest

```

## Repository Layout

* `src/` – Core processing modules (`dataloader.py`, `governance_stats.py`, `health_check.py`, etc.)
* `data/` – Raw inputs and cached GitHub data
* `output/` – Generated reports and figures (timestamped runs)
* `enhance_data/` – Data enrichment helpers (company/person merging)
* `database_schema.sql` – DB schema used for proposal SQLite files

## Outputs

Reports and visualisations are written to `output/<timestamp>/`.

**Example outputs include:**

* `combined_projects_health_report.md`
* `governance_statistics.md`
* Radar charts under `radar_charts/`


## Contributing

Follow the project's style: concise, typed Python functions, single-purpose helpers, and fail-fast checks.



## Contact

For questions about the dataset or analysis approach, open an issue in this repository.