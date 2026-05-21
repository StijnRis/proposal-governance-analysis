## Goal

Research-oriented development. Prioritize functional correctness, data reproducibility, and rapid iteration speed over complex software architectures.

## Tech Stack

* **Language:** Python 3.13
* **Environment:** `uv` managing a local `venv`.
* **Primary Libraries:** `sqlite3`, `pandas`
* **Style:** Clean functional paradigm utilizing strict explicit type hints.

## Execution

* Execute pipeline: `uv run main.py`
* Execute tests: `uv run pytest` (only core math/logic tests, no setup or integration tests).
* **Mandatory Verification:** Always run both the pipeline and execute tests immediately after code modifications. If runtime or test errors occur, debug and fix them instantly before adding new features.

---

## Evolution & Restructuring ("Addition through Removal")

When introducing new features, datasets, or logic, prioritize minimizing codebase friction:

* **Refactor First:** Do not layer new code on top of brittle architecture. Restructure, merge, or rewrite existing functions to cleanly absorb new requirements.
* **Addition through Removal:** Actively delete dead code, deprecated logic, and redundant helper functions. Keeping the codebase small is the best way to maintain velocity.
* **Synchronized Test Updates:** When changing, restructuring, or removing functions, **immediately update or delete their corresponding tests**. Never leave broken or stale tests in the codebase.
* **Zero Backward Compatibility:** Ignore historical constraints. Break existing internal APIs aggressively if it results in a cleaner, simpler codebase.

---

## AI Behavior & Coding Standards

### 1. Code Implementation

* **No Placeholders:** Write 100% complete, executable code. Never use inline placeholders like `# TODO` or `# ... logic goes here`.
* **Don't Reinvent the Wheel:** Lean heavily on libraries to process data rather than writing custom logic.

### 2. Architecture & Design

* **Modular Functions:** Write small, single-purpose, highly testable functions. Avoid complex, monolithic classes.
* **Strict Typing:** Apply precise type hints to all variables, arguments, and return types. The `Any` type is strictly forbidden.

### 3. Testing Philosophy

* **Ultra-Minimal Tests:** Write the absolute bare minimum number of tests required to prove core math, data transformations, or critical pipeline logic works. Avoid complex test suites or testing trivial setup code.
* **Keep Pace with Changes:** Treat tests as living documentation that must change instantly alongside the source code.

### 4. Execution Safety & Performance

* **Natural Propagation:** Allow exceptions to bubble up naturally to catch failures early. Avoid silent `try-except` blocks.
* **Memory Efficiency:** Write code that processes large research datasets efficiently (e.g., proper use of pandas vectors, avoiding unnecessary data duplication).

### 5. Documentation

* **Self-Documenting:** Rely on highly descriptive function and variable names.
* **Minimalist Comments:** Omit docstrings and inline comments unless explaining a non-obvious, complex mathematical algorithm.

---

## Project Structure

```text
├── .env                  # Configuration variables (e.g., DATABASE_PATH)
├── database_schema.md    # Source of truth for DB architecture
├── main.py               # Application entry point
├── src/                  # Core processing logic and pipelines
├── notebooks/            # Rapid prototyping and isolated visual exploration
├── tests/                # Minimal, high-impact unit tests for core math/logic only
└── output/               # Final generated datasets, figures, and reports

```