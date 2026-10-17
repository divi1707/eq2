"""Smoke test for the inspection report.

The data here is RANDOM TEST DATA generated only to check that the report
code runs and returns sensible structure. It is not an earthquake catalogue.
"""
import numpy as np
import pandas as pd

from src.common import load_config
from src.inspect_catalogue import build_report, max_curvature_mc


def _toy_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.to_datetime("1990-01-01") + pd.to_timedelta(rng.integers(0, 20 * 365, n), unit="D")
    return pd.DataFrame({
        "time": t.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "latitude": rng.uniform(26.5, 31.5, n),
        "longitude": rng.uniform(78, 88, n),
        "depth": rng.uniform(0, 60, n),
        "mag": np.round(rng.uniform(2.5, 6.0, n), 1),
        "magType": rng.choice(["mb", "ml", "mww"], n),
        "net": "us",
        "id": [f"t{i}" for i in range(n)],
    })


def test_report_structure_and_counts():
    df = _toy_frame()
    df.loc[0, "latitude"] = 45.0  # one point outside the bounds
    rep = build_report(df, load_config())
    assert rep["n_rows"] == 400
    assert rep["outside_study_bounds"] == 1
    assert rep["duplicate_event_ids"] == 0
    assert "magnitude_type_by_decade" in rep
    assert set(rep["events_per_year_by_min_magnitude"]) >= {"M>=2.5", "M>=4.0"}


def test_max_curvature_needs_enough_events():
    assert max_curvature_mc(np.array([3.0, 3.1])) is None
    mags = np.concatenate([np.full(200, 3.0), np.full(20, 4.0), np.full(10, 5.0)])
    assert max_curvature_mc(mags) == 3.05  # centre of the 3.0 bin


def test_missing_magnitude_type_is_reported_not_dropped():
    df = _toy_frame(60)
    df.loc[:9, "magType"] = None
    rep = build_report(df, load_config())
    assert "(missing)" in rep["magnitude_type_by_decade"]
    total = sum(sum(r.values()) for r in rep["magnitude_type_by_decade"].values())
    assert total == 60
