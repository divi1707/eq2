"""Pure data functions used by the Streamlit catalogue explorer.

Kept separate from the app so they can be unit tested. Nothing here
changes the saved catalogue. These are views for inspection only, not the
final cleaning step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.download_catalogue import detect_columns
from src.inspect_catalogue import max_curvature_mc


def prepare_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Standardise a raw USGS-style frame.

    Returns (frame, n_dropped). Rows whose time, latitude or longitude
    cannot be read are dropped and counted, never silently kept.
    Magnitude and depth may stay missing (NaN).
    """
    roles = detect_columns(list(raw.columns))
    out = pd.DataFrame({
        "id": raw[roles["event_id"]].astype(str),
        "time": pd.to_datetime(raw[roles["time"]], utc=True, errors="coerce"),
        "lat": pd.to_numeric(raw[roles["latitude"]], errors="coerce"),
        "lon": pd.to_numeric(raw[roles["longitude"]], errors="coerce"),
        "mag": pd.to_numeric(raw[roles["magnitude"]], errors="coerce"),
    })
    out["depth"] = (
        pd.to_numeric(raw[roles["depth"]], errors="coerce") if "depth" in roles else np.nan
    )
    if "mag_type" in roles:
        mt = raw[roles["mag_type"]].astype("string").str.strip().str.lower()
        out["mag_type"] = mt.fillna("(missing)").replace({"": "(missing)", "nan": "(missing)"}).astype(object)
    else:
        out["mag_type"] = "(not given)"
    before = len(out)
    out = out.dropna(subset=["time", "lat", "lon"]).reset_index(drop=True)
    return out, before - len(out)


def filter_frame(
    df: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    year_range: tuple[int, int],
    min_mag: float,
) -> tuple[pd.DataFrame, int]:
    """Apply box, year and magnitude filters.

    ``bounds`` is (min_lat, max_lat, min_lon, max_lon). Returns the filtered
    frame and the number of events in the whole frame that lie outside the
    box. Events with missing magnitude are removed only if min_mag > 0 is
    requested through the magnitude filter, and that choice is explicit.
    """
    la0, la1, lo0, lo1 = bounds
    inside = df["lat"].between(la0, la1) & df["lon"].between(lo0, lo1)
    n_outside = int((~inside).sum())
    year = df["time"].dt.year
    keep = inside & year.between(year_range[0], year_range[1]) & (df["mag"] >= min_mag)
    return df[keep].reset_index(drop=True), n_outside


def events_per_year(df: pd.DataFrame) -> pd.Series:
    """Number of events per calendar year, with zero-event years included."""
    if df.empty:
        return pd.Series(dtype="int64")
    y = df["time"].dt.year
    return y.value_counts().reindex(range(int(y.min()), int(y.max()) + 1), fill_value=0).sort_index()


def magnitude_histogram(df: pd.DataFrame, bin_width: float = 0.1) -> pd.Series:
    """Counts per magnitude bin (index is the bin centre)."""
    m = df["mag"].dropna().to_numpy()
    if len(m) == 0:
        return pd.Series(dtype="int64")
    lo = np.floor(m.min() / bin_width) * bin_width
    hi = np.ceil(m.max() / bin_width) * bin_width + bin_width
    edges = np.arange(lo, hi + bin_width / 2, bin_width)
    counts, _ = np.histogram(m, bins=edges)
    return pd.Series(counts, index=np.round(edges[:-1] + bin_width / 2, 2))


def decade_type_table(df: pd.DataFrame) -> pd.DataFrame:
    """Magnitude type counts by decade (rows: type, columns: decade)."""
    if df.empty:
        return pd.DataFrame()
    decade = (df["time"].dt.year // 10 * 10).astype(int).astype(str) + "s"
    return pd.crosstab(df["mag_type"], decade)


def completeness_table(df: pd.DataFrame, window_years: int = 5) -> pd.DataFrame:
    """Rough completeness (max curvature) per window of years.

    This is a first guide only. It is not a final completeness threshold.
    """
    if df.empty:
        return pd.DataFrame(columns=["window", "n_events", "rough_Mc"])
    year = df["time"].dt.year
    rows = []
    for y0 in range(int(year.min()) // window_years * window_years, int(year.max()) + 1, window_years):
        sel = df.loc[(year >= y0) & (year < y0 + window_years), "mag"].to_numpy()
        rows.append({
            "window": f"{y0}-{y0 + window_years - 1}",
            "n_events": int(np.isfinite(sel).sum()),
            "rough_Mc": max_curvature_mc(sel),
        })
    return pd.DataFrame(rows)
