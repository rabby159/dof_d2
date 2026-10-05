import numpy as np
import pandas as pd
from features import SEQ, TAB_COLS, forecast_inputs, training_set

CSV = "dengu_weather_dataset.csv"


def load():
    return pd.read_csv(CSV, parse_dates=["start_date"])


def test_training_set_has_no_nans():
    d, X, y, S, valid = training_set(load())
    assert len(valid) > 300
    assert not X.iloc[valid].isna().any().any()
    assert not np.isnan(y.values[valid]).any()
    assert list(X.columns) == TAB_COLS


def test_no_same_week_information_leak():
    """Features for week t must not change if week t's own cases/weather change."""
    df = load()
    d1, X1, *_ = training_set(df)
    df2 = df.copy()
    i = len(df2) - 1
    df2.loc[i, ["dengue_cases", "temp_mean", "rainfall_total", "humidity_mean"]] = [99999, 40, 900, 99]
    d2, X2, *_ = training_set(df2)
    pd.testing.assert_series_equal(X1.iloc[i], X2.iloc[i])


def test_forecast_inputs_shapes():
    X, S, nd, last = forecast_inputs(load())
    assert X.shape == (1, len(TAB_COLS)) and S.shape[0] == SEQ
    assert nd > load().start_date.max()
    assert not X.isna().any().any()


def test_forecast_runs_and_is_positive():
    from forecast import Forecaster
    r = Forecaster("models").predict(load())
    assert r["pred"] > 0 and r["low"] <= r["pred"] <= r["high"]
