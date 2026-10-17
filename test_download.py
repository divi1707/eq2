"""Tests for the download module using a MOCK session.

The CSV rows below are synthetic test fixtures made only to exercise the
parser. They are not earthquake records and are never used outside tests.
The live USGS service is NOT contacted by these tests.
"""
import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from src.common import load_config
from src.download_catalogue import (
    CatalogueError,
    NetworkUnavailable,
    detect_columns,
    download_window,
    http_get,
    ingest_local_csv,
    merge_windows,
    parse_csv_text,
    run_live_download,
)

HEADER = "time,latitude,longitude,depth,mag,magType,net,id,type\n"


def fake_rows(ids, base_time="2000-01-01T00:00:00.000Z"):
    return "".join(
        f"{base_time},28.0,84.0,10.0,4.5,mb,us,{i},earthquake\n" for i in ids
    )


class FakeResponse:
    def __init__(self, text="", status=200):
        self.text, self.status_code, self.headers, self.url = text, status, {}, "mock://url"


class FakeSession:
    """Answers /count and /query from a list of ids per call."""

    def __init__(self, count_fn, csv_fn):
        self.count_fn, self.csv_fn, self.headers, self.calls = count_fn, csv_fn, {}, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params)))
        if url.endswith("/count"):
            return FakeResponse(str(self.count_fn(params)))
        return FakeResponse(self.csv_fn(params))


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c["paths"]["raw_dir"] = str(tmp_path / "raw")
    c["catalogue"]["pause_between_requests_s"] = 0
    c["catalogue"]["retry_backoff_s"] = 0
    c["catalogue"]["max_retries"] = 2
    c["study"]["start_date"], c["study"]["end_date"] = "2000-01-01", "2004-12-31"
    return c


# ---- parsing and validation ------------------------------------------------
def test_html_error_page_is_rejected():
    with pytest.raises(CatalogueError, match="HTML"):
        parse_csv_text("<!DOCTYPE html><html><body>Service down</body></html>")


def test_empty_body_is_rejected():
    with pytest.raises(CatalogueError, match="empty"):
        parse_csv_text("   \n")


def test_missing_required_column_is_reported_with_actual_headers():
    with pytest.raises(CatalogueError, match="Actual headers"):
        parse_csv_text("foo,bar\n1,2\n")


def test_header_only_csv_gives_empty_frame():
    assert parse_csv_text(HEADER).empty


def test_column_detection_is_case_insensitive():
    m = detect_columns(["Time", "Latitude", "Longitude", "Mag", "ID", "magType"])
    assert m["time"] == "Time" and m["mag_type"] == "magType"


# ---- window download -------------------------------------------------------
def test_window_split_when_count_exceeds_limit(cfg, tmp_path):
    cfg["catalogue"]["max_events_per_request"] = 5
    state = {"next": 0}

    def count_fn(p):
        span = pd.Timestamp(p["endtime"]) - pd.Timestamp(p["starttime"])
        return 8 if span > pd.Timedelta(days=1000) else 3

    def csv_fn(p):
        n = count_fn(p)
        ids = [f"e{state['next'] + k}" for k in range(n)]
        state["next"] += n
        return HEADER + fake_rows(ids)

    out = tmp_path / "w"
    out.mkdir()
    manifest = []
    sess = FakeSession(count_fn, csv_fn)
    download_window(sess, cfg, pd.Timestamp("2000-01-01"), pd.Timestamp("2005-01-01"),
                    out, False, manifest)
    assert len(manifest) == 2 and all(m["status"] == "ok" for m in manifest)
    assert sum(m["rows_saved"] for m in manifest) == 6


def test_incomplete_download_is_detected(cfg, tmp_path):
    sess = FakeSession(lambda p: 10, lambda p: HEADER + fake_rows(["a", "b"]))
    out = tmp_path / "w"
    out.mkdir()
    with pytest.raises(CatalogueError, match="incomplete"):
        download_window(sess, cfg, pd.Timestamp("2000-01-01"), pd.Timestamp("2001-01-01"),
                        out, False, [])


def test_zero_count_window_is_recorded_as_empty(cfg, tmp_path):
    sess = FakeSession(lambda p: 0, lambda p: pytest.fail("query must not be called"))
    out = tmp_path / "w"
    out.mkdir()
    manifest = []
    download_window(sess, cfg, pd.Timestamp("2000-01-01"), pd.Timestamp("2001-01-01"),
                    out, False, manifest)
    assert manifest[0]["status"] == "empty"


def test_non_integer_count_is_rejected(cfg, tmp_path):
    sess = FakeSession(lambda p: "<html>oops</html>", lambda p: "")
    out = tmp_path / "w"
    out.mkdir()
    with pytest.raises(CatalogueError, match="non-integer"):
        download_window(sess, cfg, pd.Timestamp("2000-01-01"), pd.Timestamp("2001-01-01"),
                        out, False, [])


# ---- full run, merge and duplicates ----------------------------------------
def test_full_run_removes_repeated_ids_and_writes_metadata(cfg):
    # 5 one-year windows, each returning the same id: 4 repeats must be dropped.
    cfg["catalogue"]["window_years"] = 1
    sess = FakeSession(lambda p: 1, lambda p: HEADER + fake_rows(["same_id"]))
    csv_path, meta_path = run_live_download(cfg, session=sess)
    df = pd.read_csv(csv_path)
    meta = json.loads(meta_path.read_text())
    assert len(df) == 1
    assert meta["repeated_event_ids_removed"] == 4
    assert meta["mode"] == "live_download"
    assert "detected_column_roles" in meta and meta["raw_csv_sha256"]


def test_all_empty_catalogue_raises(cfg):
    sess = FakeSession(lambda p: 0, lambda p: "")
    with pytest.raises(CatalogueError, match="no events"):
        run_live_download(cfg, session=sess)


def test_merge_drops_alternative_rows_with_same_id(tmp_path):
    f1, f2 = tmp_path / "a.csv", tmp_path / "b.csv"
    f1.write_text(HEADER + fake_rows(["x", "y"]))
    f2.write_text(HEADER + fake_rows(["y", "z"]))
    df, dropped = merge_windows([f1, f2])
    assert dropped == 1 and sorted(df["id"]) == ["x", "y", "z"]


# ---- network failure and manual CSV ----------------------------------------
def test_connection_failure_raises_network_unavailable(cfg):
    class Dead:
        headers = {}

        def get(self, *a, **k):
            raise requests.ConnectionError("proxy refused")

    with pytest.raises(NetworkUnavailable):
        http_get(Dead(), "https://x", {}, 1, 2, 0)


def test_http_400_is_not_retried_and_shows_server_text():
    class Bad:
        headers = {}
        n = 0

        def get(self, *a, **k):
            Bad.n += 1
            return FakeResponse("Error 400: bad parameter", 400)

    with pytest.raises(CatalogueError, match="bad parameter"):
        http_get(Bad(), "https://x", {}, 1, 4, 0)
    assert Bad.n == 1


def test_manual_csv_is_ingested_and_flagged_unverified(cfg, tmp_path):
    src = tmp_path / "manual.csv"
    src.write_text(HEADER + fake_rows(["m1", "m2"]))
    csv_path, meta_path = ingest_local_csv(cfg, src)
    meta = json.loads(meta_path.read_text())
    assert meta["mode"] == "manual_csv" and "NOT verified" in meta["note"]
    assert len(pd.read_csv(csv_path)) == 2


def test_manual_csv_with_no_events_is_rejected(cfg, tmp_path):
    src = tmp_path / "empty.csv"
    src.write_text(HEADER)
    with pytest.raises(CatalogueError, match="no events"):
        ingest_local_csv(cfg, src)
