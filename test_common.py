import pandas as pd
import pytest

from src.common import (
    format_fdsn_time,
    load_config,
    make_time_windows,
    split_window,
    validate_config,
    window_query_times,
)


def test_windows_cover_range_without_gaps_or_overlap():
    w = make_time_windows("1980-01-01", "2025-12-31", 5)
    assert w[0][0] == pd.Timestamp("1980-01-01")
    assert w[-1][1] == pd.Timestamp("2026-01-01")  # end day is included
    for (_, e1), (s2, _) in zip(w, w[1:]):
        assert e1 == s2
    assert len(w) == 10  # 46 years in 5-year steps


def test_last_window_is_truncated():
    w = make_time_windows("2020-01-01", "2022-06-30", 2)
    assert w[-1][1] == pd.Timestamp("2022-07-01")
    assert len(w) == 2


def test_bad_windows_raise():
    with pytest.raises(ValueError):
        make_time_windows("2020-01-01", "2019-01-01", 5)
    with pytest.raises(ValueError):
        make_time_windows("2020-01-01", "2021-01-01", 0)


def test_split_window_halves_and_stops_at_one_day():
    a = pd.Timestamp("2000-01-01")
    b = pd.Timestamp("2000-01-11")
    (s1, e1), (s2, e2) = split_window(a, b)
    assert s1 == a and e2 == b and e1 == s2 and a < e1 < b
    assert split_window(a, a + pd.Timedelta(days=1)) is None


def test_query_times_exclude_next_window_start():
    st, et = window_query_times(pd.Timestamp("2000-01-01"), pd.Timestamp("2005-01-01"))
    assert st == "2000-01-01T00:00:00.000"
    assert et == "2004-12-31T23:59:59.999"
    assert format_fdsn_time(pd.Timestamp("2001-02-03 04:05:06.789")) == "2001-02-03T04:05:06.789"


def test_default_config_loads():
    cfg = load_config()
    assert cfg["grid"]["resolution_deg"] == 0.25
    assert cfg["sequence"]["input_months"] == 12


def test_config_rejects_bounds_not_divisible_by_grid():
    cfg = load_config()
    cfg["study"]["max_lat"] = 31.4  # 4.9 deg is not a whole number of 0.25 cells
    with pytest.raises(ValueError, match="whole number"):
        validate_config(cfg)


def test_config_rejects_download_floor_above_candidate_threshold():
    cfg = load_config()
    cfg["catalogue"]["download_min_magnitude"] = 5.0
    with pytest.raises(ValueError):
        validate_config(cfg)
