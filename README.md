# Proposal Governance Analysis

Simple scripts to extract statistics and visualizations from a proposals SQLite database.

Usage
- Add DATABASE_LOCATION to a .env file pointing to your SQLite database.
- Run: python main.py

Outputs (saved to output/)
- project_statistics.md, project_statistics.tex
- *_proposals_timeline.png, *_revisions_timeline.png
- *_author_tenure.png, *_author_activity.png, *_comments_distribution.png

Requirements
- Python 3.11+, pandas, matplotlib, python-dotenv

License
- MIT
