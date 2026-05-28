## Goal

Research-oriented development. Analysing how governance metrics evolve across multiple open source software projects by looking at their enhancement proposals.

## Tech Stack

* **Language:** Python 3.13
* **Environment:** `uv` managing a local `venv`.
* **Primary Libraries:** `sqlite3`, `pandas`
* **Style:** Clean functional paradigm utilizing strict explicit type hints.

## Execution & Verification
* Run Pipeline: `uv run main.py`
* Run Tests: `uv run pytest` (Core math/logic only; no setup/integration tests).
* Mandatory verification: Always run tests after any code modification. If any error occurs, stop and fix it before adding features.

## Coding & Architecture Standards
* Less is More: Prioritize brevity, readability, and reusability. If code can be refactored to be shorter without sacrificing clarity, or abstracted into a reusable component, do it. Eliminate boilerplate aggressively.
* Single-Purpose Functions: Every function must have exactly one distinct purpose. If a function is handling multiple responsibilities (e.g., fetching data and formatting it), split it into separate, dedicated helpers.
* Rich Type Safety: Avoid primitive obsession. Use exact, domain-specific types instead of generic primitives wherever possible (e.g., use pathlib.Path instead of str for file paths, datetime objects instead of raw strings/ints for time, etc.).
* Completeness: Write 100% executable code. No placeholders (`# TODO`, `# ...`).
* Modularity & Typing: Write small, single-purpose functions. Use strict type hints; `Any` and `object` is forbidden.
* Fail-Fast Error Handling: Never use silent `try-except` blocks. Fail explicitly and let exceptions bubble up naturally. Raise descriptive errors immediately when inputs, states, or data transformations are wrong.
* Dependencies: Lean heavily on standard/installed libraries. Use top-level imports only.
* Documentation: Rely on descriptive naming. Omit docstrings and comments unless explaining complex mathematical algorithms.

## Refactoring & Evolution
* Refactor First: Restructure or rewrite existing code to absorb new requirements cleanly. Do not layer new code on brittle architecture.
* Clean Codebase: Actively delete dead code, deprecated logic, and redundant helpers. Break internal APIs aggressively for simplicity.
* Synchronized Tests: Write the absolute bare minimum tests for core math/logic. Immediately update or delete corresponding tests when modifying source code—never leave broken or stale tests.

## Project Structure

```text
├── .env                  # Configuration variables (e.g., DATABASE_PATH)
├── database_schema.sql   # Source of truth for DB architecture
├── main.py               # Application entry point
├── src/                  # Core processing logic and pipelines
├── tests/                # Minimal, high-impact unit tests for core math/logic only
└── output/               # Final generated datasets, figures, and reports

```