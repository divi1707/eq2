"""Print and save a schema and statistics report for the raw catalogue.

Run this after downloading and send me the output before the cleaning
step is finalised. It reports facts only: the real column names, missing
values, magnitude types over time, yearly counts and a rough completeness
estimate. It does not change or filter any data.

    python inspect_catalogue.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import get_logger, load_config, resolve_path
from download_catalogue import detect_columns

log = get_logger("inspect")

THRESHOLDS = (2.5, 3.0, 3.5, 4.0, 4.5, 5.0)


def max_curvature_mc(mags: np.ndarray, bin_width: float = 0.1) -> float | None:
    """Rough completeness magnitude by the maximum-curvature method.

    Mc is the magnitude bin with the most events. This is only a first
    estimate and is known to run low, so it is reported as a guide and
    not as a final threshold. Returns None if there are too few events.
    """
    mags = mags[np.isfinite(mags)]
    if len(mags) < 50:
        return None
    lo = np.floor(mags.min() / bin_width) * bin_width
    hi = np.ceil(mags.max() / bin_width) * bin_width + bin_width
    edges = np.arange(lo, hi + bin_width / 2, bin_width)
    counts, _ = np.histogram(mags, bins=edges)
    return round(float(edges[int(np.argmax(counts))] + bin_width / 2), 2)


def build_report(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Compute the report as a plain dictionary."""
    roles = detect_columns(list(df.columns))
    s = cfg["study"]
    t = pd.to_datetime(df[roles["time"]], utc=True, errors="coerce")
    lat = pd.to_numeric(df[roles["latitude"]], errors="coerce")
    lon = pd.to_numeric(df[roles["longitude"]], errors="coerce")
    mag = pd.to_numeric(df[roles["magnitude"]], errors="coerce")

    rep: dict[str, Any] = {
        "n_rows": int(len(df)),
        "columns": {c: {"dtype": str(df[c].dtype), "n_missing": int(df[c].isna().sum())}
                    for c in df.columns},
        "detected_roles": roles,
        "time_range": [str(t.min()), str(t.max())],
        "unparseable_times": int(t.isna().sum()),
        "outside_study_bounds": int(
            (~lat.between(s["min_lat"], s["max_lat"]) | ~lon.between(s["min_lon"], s["max_lon"])).sum()
        ),
        "duplicate_event_ids": int(df.duplicated(subset=roles["event_id"]).sum()),
        "magnitude_summary": mag.describe().round(3).to_dict(),
        "missing_magnitude": int(mag.isna().sum()),
    }
    if "depth" in roles:
        dep = pd.to_numeric(df[roles["depth"]], errors="coerce")
        rep["depth_summary"] = dep.describe().round(2).to_dict()
        rep["missing_depth"] = int(dep.isna().sum())
    for role in ("event_type", "status", "network", "magnitude_source", "location_source"):
        if role in roles:
            rep[f"counts_{role}"] = df[roles[role]].value_counts(dropna=False).head(15).to_dict()

    year = t.dt.year
    frame = pd.DataFrame({"year": year, "mag": mag})
    rep["events_per_year_by_min_magnitude"] = {
        f"M>={th}": frame[frame.mag >= th].groupby("year").size().reindex(
            range(int(year.min()), int(year.max()) + 1), fill_value=0
        ).to_dict()
        for th in THRESHOLDS
    }

    if "mag_type" in roles:
        mt = (df[roles["mag_type"]].astype("string").str.strip().str.lower()
              .fillna("(missing)").replace({"": "(missing)", "nan": "(missing)"}).astype(object))
        decade = (year // 10 * 10).astype("Int64").astype(str) + "s"
        tab = pd.crosstab(mt, decade)
        rep["magnitude_type_by_decade"] = {k: {c: int(v) for c, v in row.items()}
                                           for k, row in tab.iterrows()}

    mc_windows = {}
    for y0 in range(int(year.min()) // 5 * 5, int(year.max()) + 1, 5):
        sel = mag[(year >= y0) & (year < y0 + 5)].to_numpy()
        mc_windows[f"{y0}-{y0 + 4}"] = {
            "n_events": int(np.isfinite(sel).sum()),
            "mc_max_curvature": max_curvature_mc(sel),
        }
    rep["rough_completeness_by_5yr_window"] = mc_windows
    return rep


def print_report(rep: dict[str, Any]) -> None:
    print("\n=== CATALOGUE REPORT ===")
    print(f"Rows: {rep['n_rows']}   Time range: {rep['time_range'][0]}  ->  {rep['time_range'][1]}")
    print("\nColumns (name | dtype | missing):")
    for c, v in rep["columns"].items():
        print(f"  {c:<18} {v['dtype']:<10} {v['n_missing']}")
    print("\nDetected roles:", rep["detected_roles"])
    print(f"\nUnparseable times: {rep['unparseable_times']}   "
          f"Outside study bounds: {rep['outside_study_bounds']}   "
          f"Duplicate ids: {rep['duplicate_event_ids']}   "
          f"Missing magnitude: {rep['missing_magnitude']}")
    print("\nMagnitude summary:", rep["magnitude_summary"])
    if "depth_summary" in rep:
        print("Depth summary:", rep["depth_summary"], " missing:", rep["missing_depth"])
    for k, v in rep.items():
        if k.startswith("counts_"):
            print(f"\n{k}: {v}")
    if "magnitude_type_by_decade" in rep:
        print("\nMagnitude type by decade:")
        for k, row in rep["magnitude_type_by_decade"].items():
            print(f"  {k:<10} {row}")
    print("\nRough completeness (max curvature) per 5-year window:")
    for w, v in rep["rough_completeness_by_5yr_window"].items():
        print(f"  {w}: n={v['n_events']:<6} Mc~{v['mc_max_curvature']}")
    print("\nEvents per 5-year block at each minimum magnitude:")
    ey = rep["events_per_year_by_min_magnitude"]
    years = sorted(next(iter(ey.values())).keys())
    blocks = sorted({y // 5 * 5 for y in years})
    print("  block      " + "  ".join(f"{k:>7}" for k in ey))
    for b in blocks:
        row = [sum(v for y, v in ey[k].items() if b <= y < b + 5) for k in ey]
        print(f"  {b}-{b + 4}  " + "  ".join(f"{x:>7}" for x in row))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Inspect the raw catalogue.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--csv", help="Override the raw CSV path.")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    csv = Path(args.csv) if args.csv else resolve_path(cfg, "raw_dir") / "usgs_catalogue_raw.csv"
    if not csv.exists():
        log.error("Raw catalogue not found at %s. Run download_catalogue.py first.", csv)
        return 1
    df = pd.read_csv(csv, low_memory=False)
    if df.empty:
        log.error("The raw catalogue is empty.")
        return 1
    rep = build_report(df, cfg)
    print_report(rep)
    out = resolve_path(cfg, "metrics_dir")
    out.mkdir(parents=True, exist_ok=True)
    (out / "catalogue_inspection.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    log.info("Saved %s", out / "catalogue_inspection.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
