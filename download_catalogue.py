"""Download the earthquake catalogue from the USGS ComCat FDSN event service.

What this module does
---------------------
* Splits the study period into time windows (config: catalogue.window_years).
* For each window, asks the service how many events match (``/count``).
  If the count exceeds the service limit, the window is split in half
  until each piece fits.
* Downloads each window as CSV, checks it (not an HTML error page, has the
  expected columns, row count matches the reported count) and saves it.
* Merges the windows, drops repeated event ids, and writes one raw CSV plus
  a JSON metadata file describing exactly how the data was obtained.
* If the live service cannot be reached, it can ingest a CSV that you
  downloaded yourself (``--from-csv``). It never invents data.

Run from the folder holding these files:
    python download_catalogue.py --config config.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from common import (
    PROJECT_ROOT,
    get_logger,
    load_config,
    make_time_windows,
    resolve_path,
    split_window,
    window_query_times,
)

log = get_logger("download")

# Column names are matched case-insensitively against these candidates.
# The real headers are always reported, never assumed.
COLUMN_CANDIDATES: dict[str, list[str]] = {
    "time": ["time", "origintime", "origin_time"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon"],
    "depth": ["depth", "depth_km"],
    "magnitude": ["mag", "magnitude"],
    "mag_type": ["magtype", "mag_type", "magnitude_type"],
    "event_id": ["id", "eventid", "event_id"],
    "network": ["net", "network", "agency", "author"],
    "location_source": ["locationsource"],
    "magnitude_source": ["magsource"],
    "event_type": ["type", "eventtype"],
    "status": ["status"],
}
REQUIRED_ROLES = ("time", "latitude", "longitude", "magnitude", "event_id")


class CatalogueError(RuntimeError):
    """The data received is invalid, incomplete or empty."""


class NetworkUnavailable(RuntimeError):
    """The service could not be reached at all."""


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------
def detect_columns(columns: list[str]) -> dict[str, str]:
    """Map logical roles (time, latitude, ...) to the real column names."""
    lowered = {c.strip().lower(): c for c in columns}
    mapping: dict[str, str] = {}
    for role, candidates in COLUMN_CANDIDATES.items():
        for cand in candidates:
            if cand in lowered:
                mapping[role] = lowered[cand]
                break
    missing = [r for r in REQUIRED_ROLES if r not in mapping]
    if missing:
        raise CatalogueError(
            f"Required columns not found for roles {missing}. "
            f"Actual headers: {list(columns)}"
        )
    return mapping


def parse_csv_text(text: str, where: str = "response") -> pd.DataFrame:
    """Validate raw CSV text and return it as a DataFrame.

    Raises CatalogueError for HTML pages, empty bodies, malformed CSV and
    missing required columns. A header-only CSV returns an empty frame.
    """
    stripped = text.lstrip()
    if not stripped:
        raise CatalogueError(f"{where}: empty response body.")
    head = stripped[:300].lower()
    if head.startswith("<!doctype") or head.startswith("<html") or "<body" in head:
        raise CatalogueError(
            f"{where}: received an HTML page instead of CSV "
            f"(first 120 chars: {stripped[:120]!r})."
        )
    first_line = stripped.splitlines()[0]
    if "," not in first_line:
        raise CatalogueError(f"{where}: first line is not a CSV header: {first_line[:120]!r}")
    try:
        df = pd.read_csv(io.StringIO(text), low_memory=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise CatalogueError(f"{where}: malformed CSV ({exc}).") from exc
    detect_columns(list(df.columns))
    return df


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------
def http_get(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    timeout: float,
    retries: int,
    backoff: float,
) -> requests.Response:
    """GET with retries for transient failures.

    Retries connection errors, timeouts, 429 and 5xx. A 4xx error (other
    than 429) is a bad request and is raised at once with the server text.
    """
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            log.warning("Attempt %d/%d failed: %s", attempt, retries, exc)
        else:
            if resp.status_code in (200, 204):
                return resp
            if resp.status_code == 429 or resp.status_code >= 500:
                last_exc = CatalogueError(f"HTTP {resp.status_code} from service")
                log.warning("Attempt %d/%d: HTTP %d", attempt, retries, resp.status_code)
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    time.sleep(min(int(retry_after), 120))
            else:
                snippet = resp.text[:300].replace("\n", " ")
                raise CatalogueError(f"HTTP {resp.status_code} for {resp.url}: {snippet}")
        if attempt < retries:
            time.sleep(backoff * attempt)
    if isinstance(last_exc, (requests.ConnectionError, requests.Timeout)):
        raise NetworkUnavailable(f"Could not reach {url}: {last_exc}") from last_exc
    raise CatalogueError(f"Service kept failing for {url}: {last_exc}")


def build_params(cfg: dict[str, Any], start: pd.Timestamp, end_excl: pd.Timestamp) -> dict[str, Any]:
    """Build the FDSN query parameters for one half-open time window."""
    st, et = window_query_times(start, end_excl)
    s, c = cfg["study"], cfg["catalogue"]
    return {
        "starttime": st,
        "endtime": et,
        "minlatitude": s["min_lat"],
        "maxlatitude": s["max_lat"],
        "minlongitude": s["min_lon"],
        "maxlongitude": s["max_lon"],
        "minmagnitude": c["download_min_magnitude"],
        "eventtype": c["event_type"],
        "orderby": "time-asc",
    }


def fetch_count(session: requests.Session, cfg: dict[str, Any], params: dict[str, Any]) -> int:
    """Ask the service how many events match the query."""
    c = cfg["catalogue"]
    resp = http_get(
        session, f"{c['base_url']}/count", params,
        c["request_timeout_s"], c["max_retries"], c["retry_backoff_s"],
    )
    text = resp.text.strip()
    if resp.status_code == 204 or text == "":
        return 0
    try:
        return int(text)
    except ValueError as exc:
        raise CatalogueError(f"Count endpoint returned a non-integer: {text[:120]!r}") from exc


# --------------------------------------------------------------------------
# Window download
# --------------------------------------------------------------------------
def _window_filename(start: pd.Timestamp, end_excl: pd.Timestamp) -> str:
    return f"usgs_{start:%Y%m%dT%H%M%S}_{end_excl:%Y%m%dT%H%M%S}.csv"


def download_window(
    session: requests.Session,
    cfg: dict[str, Any],
    start: pd.Timestamp,
    end_excl: pd.Timestamp,
    out_dir: Path,
    force: bool,
    manifest: list[dict[str, Any]],
) -> None:
    """Download one window, splitting it if it holds too many events.

    Appends one record per saved (or empty) window to ``manifest``.
    """
    c = cfg["catalogue"]
    params = build_params(cfg, start, end_excl)
    n_reported = fetch_count(session, cfg, params)
    label = f"{start:%Y-%m-%d} to {end_excl:%Y-%m-%d}"

    if n_reported == 0:
        log.info("%s: service reports 0 events (empty window)", label)
        manifest.append({
            "window_start": str(start), "window_end_exclusive": str(end_excl),
            "count_reported": 0, "rows_saved": 0, "file": None, "sha256": None,
            "status": "empty",
        })
        return

    if n_reported > c["max_events_per_request"]:
        halves = split_window(start, end_excl)
        if halves is None:
            raise CatalogueError(
                f"{label}: {n_reported} events exceed the limit but the window "
                "cannot be split further. Raise download_min_magnitude."
            )
        log.info("%s: %d events exceed the limit, splitting window", label, n_reported)
        for s2, e2 in halves:
            download_window(session, cfg, s2, e2, out_dir, force, manifest)
        return

    path = out_dir / _window_filename(start, end_excl)
    cached = False
    if path.exists() and not force:
        try:
            df = parse_csv_text(path.read_text(encoding="utf-8"), where=path.name)
            if len(df) == n_reported:
                cached = True
                log.info("%s: using cached file (%d rows)", label, len(df))
        except CatalogueError:
            log.warning("%s: cached file is invalid, downloading again", label)

    if not cached:
        resp = http_get(
            session, f"{c['base_url']}/query", {**params, "format": "csv"},
            c["request_timeout_s"], c["max_retries"], c["retry_backoff_s"],
        )
        df = parse_csv_text(resp.text, where=label)
        if len(df) < n_reported:
            raise CatalogueError(
                f"{label}: incomplete download. Service reported {n_reported} "
                f"events but the CSV has {len(df)} rows."
            )
        if len(df) > n_reported:
            log.warning(
                "%s: CSV has %d rows, count said %d (catalogue changed between "
                "calls). Keeping all rows.", label, len(df), n_reported,
            )
        path.write_text(resp.text, encoding="utf-8")
        log.info("%s: saved %d rows", label, len(df))
        time.sleep(c["pause_between_requests_s"])

    manifest.append({
        "window_start": str(start), "window_end_exclusive": str(end_excl),
        "count_reported": n_reported, "rows_saved": int(len(df)),
        "file": str(path.relative_to(out_dir.parent)) if out_dir.parent in path.parents else path.name,
        "sha256": sha256_of(path), "status": "ok",
    })


# --------------------------------------------------------------------------
# Merge and metadata
# --------------------------------------------------------------------------
def merge_windows(files: list[Path]) -> tuple[pd.DataFrame, int]:
    """Concatenate window CSVs and drop repeated event ids.

    Returns the merged frame and the number of repeated rows removed.
    """
    frames = [parse_csv_text(f.read_text(encoding="utf-8"), where=f.name) for f in files]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise CatalogueError("All windows were empty: the catalogue has no events.")
    df = pd.concat(frames, ignore_index=True)
    id_col = detect_columns(list(df.columns))["event_id"]
    before = len(df)
    df = df.drop_duplicates(subset=id_col, keep="first")
    return df, before - len(df)


def write_raw_catalogue(
    df: pd.DataFrame, raw_dir: Path, meta: dict[str, Any], dropped_duplicates: int
) -> tuple[Path, Path]:
    """Write the merged raw CSV and its metadata JSON."""
    mapping = detect_columns(list(df.columns))
    df = df.sort_values(mapping["time"], kind="stable").reset_index(drop=True)
    csv_path = raw_dir / "usgs_catalogue_raw.csv"
    meta_path = raw_dir / "usgs_catalogue_raw.meta.json"
    df.to_csv(csv_path, index=False)
    meta.update({
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "repeated_event_ids_removed": int(dropped_duplicates),
        "columns": list(df.columns),
        "detected_column_roles": mapping,
        "raw_csv_sha256": sha256_of(csv_path),
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return csv_path, meta_path


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------
def run_live_download(cfg: dict[str, Any], force: bool = False,
                      session: requests.Session | None = None) -> tuple[Path, Path]:
    """Download the whole study period from the live service."""
    c, s = cfg["catalogue"], cfg["study"]
    raw_dir = resolve_path(cfg, "raw_dir")
    out_dir = raw_dir / "usgs_windows"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = session or requests.Session()
    session.headers.update({"User-Agent": "himalayan-convlstm-student-project/0.1"})

    windows = make_time_windows(s["start_date"], s["end_date"], c["window_years"])
    log.info("Downloading %d windows from %s", len(windows), c["base_url"])
    manifest: list[dict[str, Any]] = []
    for i, (w0, w1) in enumerate(windows, 1):
        log.info("Window %d/%d", i, len(windows))
        download_window(session, cfg, w0, w1, out_dir, force, manifest)

    files = [out_dir / Path(m["file"]).name for m in manifest if m["status"] == "ok"]
    df, dropped = merge_windows(files)
    meta = {
        "source": c["source"],
        "mode": "live_download",
        "base_url": c["base_url"],
        "query_filters": {
            "bounds": [s["min_lat"], s["max_lat"], s["min_lon"], s["max_lon"]],
            "start_date": s["start_date"], "end_date": s["end_date"],
            "minmagnitude": c["download_min_magnitude"], "eventtype": c["event_type"],
        },
        "windows": manifest,
    }
    csv_path, meta_path = write_raw_catalogue(df, raw_dir, meta, dropped)
    log.info("Saved %d events to %s", len(df), csv_path)
    return csv_path, meta_path


def ingest_local_csv(cfg: dict[str, Any], csv_file: Path) -> tuple[Path, Path]:
    """Use a USGS-format CSV that was downloaded manually.

    The file is validated but its completeness cannot be verified against
    the service, so this is recorded in the metadata.
    """
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")
    df = parse_csv_text(csv_file.read_text(encoding="utf-8"), where=csv_file.name)
    if df.empty:
        raise CatalogueError(f"{csv_file.name}: the CSV has a header but no events.")
    id_col = detect_columns(list(df.columns))["event_id"]
    before = len(df)
    df = df.drop_duplicates(subset=id_col, keep="first")
    raw_dir = resolve_path(cfg, "raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": cfg["catalogue"]["source"],
        "mode": "manual_csv",
        "original_file": str(csv_file),
        "original_sha256": sha256_of(csv_file),
        "note": "Provided by the user. Completeness against the live count "
                "endpoint was NOT verified.",
    }
    csv_path, meta_path = write_raw_catalogue(df, raw_dir, meta, before - len(df))
    log.warning("Using a MANUALLY supplied CSV (%d events). Not verified against the service.", len(df))
    return csv_path, meta_path


def print_manual_urls(cfg: dict[str, Any]) -> None:
    """Print one browser-ready query URL per window for manual download."""
    c, s = cfg["catalogue"], cfg["study"]
    for w0, w1 in make_time_windows(s["start_date"], s["end_date"], c["window_years"]):
        params = build_params(cfg, w0, w1)
        q = "&".join(f"{k}={v}" for k, v in {**params, "format": "csv"}.items())
        print(f"{c['base_url']}/query?{q}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download the USGS catalogue.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--from-csv", help="Use a USGS-format CSV you downloaded yourself.")
    p.add_argument("--print-urls", action="store_true",
                   help="Print browser URLs for manual download, then exit.")
    p.add_argument("--force", action="store_true", help="Ignore cached window files.")
    args = p.parse_args(argv)

    cfg = load_config(args.config)

    if args.print_urls:
        print_manual_urls(cfg)
        return 0
    try:
        if args.from_csv:
            ingest_local_csv(cfg, Path(args.from_csv))
        else:
            try:
                run_live_download(cfg, force=args.force)
            except NetworkUnavailable as exc:
                fb = Path(cfg["catalogue"]["local_csv_fallback"])
                fb = fb if fb.is_absolute() else PROJECT_ROOT / fb
                log.error("Live service unreachable: %s", exc)
                if fb.exists():
                    log.warning("Falling back to the manual CSV at %s", fb)
                    ingest_local_csv(cfg, fb)
                else:
                    log.error(
                        "No fallback file at %s. Run again with --print-urls, "
                        "download the files in a browser, merge them into one CSV "
                        "and pass it with --from-csv.", fb)
                    return 2
    except (CatalogueError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
