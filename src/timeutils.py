"""Shared datetime parsing utilities.

Provides a single robust parser for datetime-like series that handles:
- ISO datetimes
- numeric epoch values (seconds, milliseconds, nanoseconds)

This module centralises parsing logic so callers across the project behave
consistently.
"""

from typing import Any

import pandas as pd


def to_naive_series(series: pd.Series) -> pd.Series:
    """Parse a pandas Series into naive (tz-unaware) UTC datetimes.

    Handles:
    - ISO formatted strings
    - integer/float epoch timestamps in seconds, milliseconds, or nanoseconds
    - pandas Timestamp objects

    Returns a pandas Series of dtype datetime64[ns] (naive UTC datetimes).
    """
    # first try pandas' flexible parser
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    # if everything parsed OK, validate plausible year range; otherwise fall back
    if not parsed.isna().any():
        years = parsed.dt.year.dropna()
        # if parsed years look implausible (e.g. all near 1970 due to ns interpretation),
        # fall through to robust per-item parsing instead of accepting them.
        if not years.empty and years.max() >= 1975:
            return parsed.dt.tz_convert("UTC").dt.tz_localize(None)

    # robust per-item parsing to handle numeric epoch values (seconds or milliseconds)
    def parse_item(x: Any):
        if pd.isna(x):
            return pd.NaT
        # already parsed by pandas?
        if isinstance(x, pd.Timestamp):
            return x
        # decode bytes
        if isinstance(x, (bytes, bytearray)):
            try:
                x = x.decode("utf-8")
            except Exception:
                return pd.NaT
        # normalize strings that may have surrounding quotes or parentheses
        if isinstance(x, str):
            s = x.strip()
            # strip surrounding quotes or parentheses added by some dumpers
            if (
                (s.startswith('"') and s.endswith('"'))
                or (s.startswith("'") and s.endswith("'"))
                or (s.startswith("(") and s.endswith(")"))
            ):
                s = s[1:-1].strip()
            x = s
        # numeric types or digit-strings -> interpret as epoch
        try:
            if isinstance(x, (int, float)) or (
                isinstance(x, str)
                and x.strip().lstrip("-").replace(".", "", 1).isdigit()
            ):
                # allow floats (fractional seconds)
                val = float(x)
                # Try multiple units and pick the most plausible datetime
                candidates = []
                now_year = pd.Timestamp.utcnow().year
                # seconds
                try:
                    dt_s = pd.to_datetime(val, unit="s", utc=True, errors="coerce")
                    if not pd.isna(dt_s):
                        candidates.append((dt_s, abs(dt_s.year - now_year)))
                except Exception:
                    pass
                # milliseconds
                try:
                    dt_ms = pd.to_datetime(
                        int(val), unit="ms", utc=True, errors="coerce"
                    )
                    if not pd.isna(dt_ms):
                        candidates.append((dt_ms, abs(dt_ms.year - now_year)))
                except Exception:
                    pass
                # nanoseconds
                try:
                    dt_ns = pd.to_datetime(
                        int(val), unit="ns", utc=True, errors="coerce"
                    )
                    if not pd.isna(dt_ns):
                        candidates.append((dt_ns, abs(dt_ns.year - now_year)))
                except Exception:
                    pass

                # choose candidate closest to current year and within sensible bounds
                if candidates:
                    # filter to reasonable range (1970..now+1)
                    filtered = [
                        c for c in candidates if 1970 <= c[0].year <= now_year + 1
                    ]
                    pick_pool = filtered if filtered else candidates
                    pick = min(pick_pool, key=lambda t: t[1])[0]
                    return pick
        except Exception:
            pass

        # fallback to pandas parser
        try:
            return pd.to_datetime(x, utc=True, errors="coerce")
        except Exception:
            return pd.NaT

    parsed_items = series.apply(parse_item)

    # convert any pandas.Timestamp to naive datetimes (drop tz)
    def to_naive(ts: Any):
        if pd.isna(ts):
            return pd.NaT
        try:
            ts_utc = (
                ts.tz_convert("UTC") if getattr(ts, "tzinfo", None) is not None else ts
            )
            py = ts_utc.to_pydatetime()
            return py.replace(tzinfo=None)
        except Exception:
            try:
                return (
                    pd.to_datetime(ts, utc=True, errors="coerce")
                    .to_pydatetime()
                    .replace(tzinfo=None)
                )
            except Exception:
                return pd.NaT

    naive = parsed_items.apply(lambda ts: to_naive(ts))
    return pd.to_datetime(naive, errors="coerce")
