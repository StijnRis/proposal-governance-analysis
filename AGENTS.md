# Project Agent Rules: Research Mode

## Goal
Research-oriented development. Prioritize functional correctness, reproducibility, and speed of iteration over complex abstractions. 

## Tech Stack
- Language: Python 3.13 with uv and venv for environment management.
- Primary Libraries: sqlite3, pandas
- Style: Functional approach with explicit type hints.

## AI Behavior Guidelines
- Modular Code: Prefer small, testable functions over large, monolithic classes.
- No Placeholders: Provide complete code implementations. Do not use comments like "# ... logic goes here".
- Keep it simple: Avoid over-engineering. Use straightforward solutions that get the job done efficiently.
- Documentation: Focus on clear code. Use comments only when necessary to explain non-obvious logic, do not over-document.
- Use library, library functions and built-in features as much as possibleto handle tasks instead of reinventing the wheel.
- Run project at least once after code generation to ensure it works as expected. If errors occur, debug and fix them before proceeding with further development.

## Coding Standards
- Documentation: Use descriptive function names. Use docstrings only for complex algorithmic logic.
- Error Handling: Let exceptions propagate naturally. Avoid try-except blocks unless necessary for critical operations.
- Data Handling: Prioritize memory efficiency when dealing with large research datasets.

## Project Structure'
- /output: Final results, visualizations, and reports.
- /src: Core logic and processing scripts.
- /notebooks: Rapid prototyping and visualization.
- /tests: Only critical unit tests for core functions, no complicated test suites.
- .env file for configuration, such as database paths.
- See database_schema.md for database structure details.