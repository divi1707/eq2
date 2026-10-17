"""Tests for the explorer's data functions.

The frames are SYNTHETIC TEST DATA made only to check the logic. They are
not earthquake records.
"""
import numpy as np
import pandas as pd
import pytest

from src.explore import (
    completeness_table,
    decade_type_table,
    events_per_year,
    filter_frame,
    magnitude_histogram,
    prepare_frame,
)


def raw_frame():
    return pd.DataFrame({
        "time": ["1995-03-01T00:00:00.000Z", "1995-06-01T00:00:00.000Z",
                 "2001-01-01T00:00:00.000Z", "not a time"],
        "latitude": [28.0, 40.0, 29.0, 28.0],
        "longitude": [84.0, 84.0, 80.0, 84.0],
        "depth": [10.0, 5.0, np.nan, 3.0],
        "mag": [4.0, 5.0, np.nan, 4.5],
        "magType": ["mb", "MWW", None, "mb"],
        "id": ["a", "b", "c", "d"],
    })


def test_prepare_drops_unreadable_rows_and_keeps_missing_mag():
    df, dropped = prepare_frame(raw_frame())
    assert dropped == 1 and len(df) == 3
    assert df["mag"].isna().sum() == 1  # missing stays missing, not zero
    assert set(df["mag_type"]) == {"mb", "mww", "(missing)"}


def test_filter_counts_outside_box_and_applies_filters():
    df, _ = prepare_frame(raw_frame())
    view, outside = filter_frame(df, (26.5, 31.5, 78.0, 88.0), (1990, 2000), 0.0)
    assert outside == 1  # the 40 N event
    assert list(view["id"]) == ["a"]  # b is outside the box, c is 2001 with no magnitude


def test_missing_magnitude_is_excluded_by_magnitude_filter():
    df, _ = prepare_frame(raw_frame())
    view, _ = filter_frame(df, (26.5, 31.5, 78.0, 88.0), (1990, 2010), 0.0)
    assert "c" not in set(view["id"])  # NaN >= 0 is False: removal is explicit


def test_events_per_year_includes_zero_years():
    df, _ = prepare_frame(raw_frame())
    s = events_per_year(df)
    assert s.loc[1995] == 2 and s.loc[1998] == 0 and s.loc[2001] == 1


def test_histogram_and_tables_run():
    df, _ = prepare_frame(raw_frame())
    assert magnitude_histogram(df).sum() == 2
    assert not decade_type_table(df).empty
    assert list(completeness_table(df).columns) == ["window", "n_events", "rough_Mc"]


def test_empty_inputs_do_not_crash():
    empty, _ = prepare_frame(raw_frame())
    empty = empty.iloc[0:0]
    assert events_per_year(empty).empty
    assert magnitude_histogram(empty).empty
    assert decade_type_table(empty).empty
    assert completeness_table(empty).empty
