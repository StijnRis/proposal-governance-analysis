import csv
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from timeutils import to_naive_series

DB = "data/dataset.sqlite"
PROJECT_ID = 5
OUT = Path("output") / f"{PROJECT_ID}_rust" / "stage_parsed_samples.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(DB)
cur = conn.cursor()
cur.execute(
    "SELECT proposal_id, normalised_status, created_at FROM StageHistory WHERE project_id = ? ORDER BY proposal_id LIMIT 500",
    (PROJECT_ID,),
)
rows = cur.fetchall()
conn.close()

# collect raw values and parsed values
raw_vals = [r[2] for r in rows]
import pandas as pd

series = pd.Series(raw_vals)
parsed = to_naive_series(series)

with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["proposal_id", "normalised_status", "raw_created_at", "parsed_created_at"]
    )
    for (pid, status, raw), p in zip(rows, parsed.tolist()):
        writer.writerow([pid, status, raw, p.isoformat() if pd.notna(p) else ""])

print(f"Wrote samples to {OUT}")
