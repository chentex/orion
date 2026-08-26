# pylint: disable=protected-access
"""Tests for CMR direction filtering in _analyze()."""

import pandas as pd
import pytest
from orion.algorithms.cmr.cmr import CMR


def _make_test_config():
    return {
        "name": "test-cmr",
        "uuid_field": "uuid",
        "version_field": "ocpVersion",
    }


def _make_metrics_config():
    return {
        "metric_up": {
            "direction": 1,
            "labels": [],
            "threshold": 0,
            "correlation": "",
            "context": None,
        },
        "metric_down": {
            "direction": -1,
            "labels": [],
            "threshold": 0,
            "correlation": "",
            "context": None,
        },
    }


def _make_dataframe():
    return pd.DataFrame({
        "uuid": ["uuid-1", "uuid-2"],
        "ocpVersion": ["4.19", "4.20"],
        "timestamp": [1700000000, 1700100000],
        "buildUrl": ["http://build1", "http://build2"],
        "metric_up": [100.0, 50.0],
        "metric_down": [100.0, 50.0],
    })


def test_cmr_filters_wrong_direction():
    """CMR should filter changepoints that go opposite to configured direction.

    metric_up has direction=1 (higher is worse). mean_2=50 < mean_1=100 is an
    improvement, not a regression. It should be filtered out.

    metric_down has direction=-1 (lower is worse). mean_2=50 < mean_1=100 is a
    regression. It should remain.
    """
    df = _make_dataframe()
    test = _make_test_config()
    metrics_config = _make_metrics_config()
    options = {"ackMap": None, "collapse": False}

    algorithm = CMR(
        dataframe=df,
        test=test,
        options=options,
        metrics_config=metrics_config,
    )

    _, change_points = algorithm._analyze()

    # metric_up: direction=1, value dropped (100->50) = improvement. Should be filtered.
    assert len(change_points.get("metric_up", [])) == 0, \
        "metric_up changepoint should be filtered (direction=1, value decreased)"

    # metric_down: direction=-1, value dropped (100->50) = regression. Should remain.
    assert len(change_points.get("metric_down", [])) == 1, \
        "metric_down changepoint should remain (direction=-1, value decreased)"


def test_cmr_preserves_metadata_columns():
    """CMR should average metrics while preserving metadata from the
    most recent baseline run."""
    metrics_config = {
        "metric_up": {
            "direction": 1, "labels": [], "threshold": 0,
            "correlation": "", "context": None,
        },
    }
    df = pd.DataFrame({
        "uuid": ["uuid-1", "uuid-2", "uuid-3", "uuid-4"],
        "ocpVersion": ["4.17", "4.18", "4.19", "4.20"],
        "timestamp": [1700000000, 1700100000, 1700200000, 1700300000],
        "buildUrl": [
            "http://build1",
            "http://build2",
            "http://build3",
            "http://build4",
        ],
        "metric_up": [10.0, 10.5, 9.8, 20.0],
    })
    test = _make_test_config()
    options = {"ackMap": None, "collapse": False}
    algorithm = CMR(
        dataframe=df, test=test, options=options,
        metrics_config=metrics_config,
    )
    result = algorithm.combine_and_average_runs(df)
    assert len(result) == 2

    baseline = result.iloc[0]
    assert baseline["uuid"] == "uuid-3"
    assert baseline["ocpVersion"] == "4.19"
    assert baseline["timestamp"] == 1700200000
    assert baseline["buildUrl"] == "http://build3"
    assert baseline["metric_up"] == pytest.approx(10.1)
    assert pd.api.types.is_numeric_dtype(result["timestamp"])

    current = result.iloc[1]
    assert current["uuid"] == "uuid-4"
    assert current["ocpVersion"] == "4.20"


def test_cmr_keeps_direction_zero():
    """CMR with direction=0 should keep all changepoints regardless of direction."""
    metrics_config = {
        "metric_neutral": {
            "direction": 0,
            "labels": [],
            "threshold": 0,
            "correlation": "",
            "context": None,
        },
    }
    df = pd.DataFrame({
        "uuid": ["uuid-1", "uuid-2"],
        "ocpVersion": ["4.19", "4.20"],
        "timestamp": [1700000000, 1700100000],
        "buildUrl": ["http://build1", "http://build2"],
        "metric_neutral": [100.0, 50.0],
    })
    test = _make_test_config()
    options = {"ackMap": None, "collapse": False}

    algorithm = CMR(
        dataframe=df,
        test=test,
        options=options,
        metrics_config=metrics_config,
    )

    _, change_points = algorithm._analyze()

    assert len(change_points.get("metric_neutral", [])) == 1, \
        "direction=0 should keep all changepoints"
