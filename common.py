"""Shared helpers: config loading, validation, date windows, logging.

Nothing in this module touches the network or any catalogue data.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


def get_logger(name: str) -> logging.Logger:
    """Return a logger that prints timestamped messages to the terminal."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the YAML configuration.

    A relative path is tried against the current directory first, then
    against the project root, so the scripts work from any folder.
    """
    if path is None:
        cfg_path = PROJECT_ROOT / "config.yaml"
    else:
        cfg_path = Path(path)
        if not cfg_path.is_absolute() and not cfg_path.exists():
            cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """Raise ValueError with a clear message if the config is unusable."""
    for section in ("study", "catalogue", "paths", "grid", "sequence"):
        if section not in cfg:
            raise ValueError(f"Config is missing the '{section}' section.")

    s = cfg["study"]
    if not (-90 <= s["min_lat"] < s["max_lat"] <= 90):
        raise ValueError("study: need -90 <= min_lat < max_lat <= 90.")
    if not (-180 <= s["min_lon"] < s["max_lon"] <= 180):
        raise ValueError("study: need -180 <= min_lon < max_lon <= 180.")

    start, end = pd.Timestamp(s["start_date"]), pd.Timestamp(s["end_date"])
    if end < start:
        raise ValueError("study: end_date is before start_date.")

    res = float(cfg["grid"]["resolution_deg"])
    if res <= 0:
        raise ValueError("grid: resolution_deg must be positive.")
    for name, span in (
        ("latitude", s["max_lat"] - s["min_lat"]),
        ("longitude", s["max_lon"] - s["min_lon"]),
    ):
        n = span / res
        if abs(n - round(n)) > 1e-9:
            raise ValueError(
                f"grid: the {name} span ({span} deg) is not a whole number of "
                f"{res} deg cells. Adjust the bounds or the resolution."
            )

    c = cfg["catalogue"]
    if c["download_min_magnitude"] > c["candidate_min_magnitude"]:
        raise ValueError(
            "catalogue: download_min_magnitude must not exceed "
            "candidate_min_magnitude, otherwise completeness cannot be studied."
        )
    if c["window_years"] < 1:
        raise ValueError("catalogue: window_years must be at least 1.")


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a path from cfg['paths'] against the project root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else PROJECT_ROOT / p


def make_time_windows(
    start: str | pd.Timestamp, end_inclusive: str | pd.Timestamp, years: int
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split a date range into half-open windows [start, end).

    ``end_inclusive`` is the last calendar day to include, so the final
    window ends at the start of the following day. Windows never overlap
    and together cover the range exactly.
    """
    if years < 1:
        raise ValueError("years must be at least 1.")
    s = pd.Timestamp(start)
    e = pd.Timestamp(end_inclusive) + pd.Timedelta(days=1)
    if e <= s:
        raise ValueError("end date must not be before start date.")
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = s
    while cur < e:
        nxt = min(cur + pd.DateOffset(years=years), e)
        windows.append((cur, nxt))
        cur = nxt
    return windows


def split_window(
    start: pd.Timestamp, end: pd.Timestamp
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], tuple[pd.Timestamp, pd.Timestamp]] | None:
    """Split a half-open window in two at its midpoint.

    Returns None when the window is too short to split (one day or less).
    """
    if end - start <= pd.Timedelta(days=1):
        return None
    mid = (start + (end - start) / 2).floor("s")
    if not (start < mid < end):
        return None
    return (start, mid), (mid, end)


def format_fdsn_time(ts: pd.Timestamp) -> str:
    """Format a timestamp as ISO 8601 with milliseconds (no timezone)."""
    return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}"


def window_query_times(start: pd.Timestamp, end_exclusive: pd.Timestamp) -> tuple[str, str]:
    """Return (starttime, endtime) strings for a half-open window.

    The service treats endtime as inclusive, so one millisecond is removed
    from the exclusive end to stop events being fetched by two windows.
    """
    return format_fdsn_time(start), format_fdsn_time(end_exclusive - pd.Timedelta(milliseconds=1))
