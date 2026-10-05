"""Shared feature engineering for training, evaluation and the Streamlit app.

Task definition (true 1-week-ahead forecast):
    Predict weekly dengue cases for week t using ONLY information available up to week t-1
    (past cases, past weather) plus the calendar position of week t.
    Target = log growth:  log(1+cases_t) - log(1+cases_{t-1}).
"""
import numpy as np
import pandas as pd

SEQ = 8                                   # LSTM window length (weeks)
WEATHER = ["temp_mean", "rainfall_total", "humidity_mean"]
SEQ_COLS = ["lc"] + WEATHER + ["sin_w", "cos_w"]
RAW_COLS = ["start_date", "dengue_cases"] + WEATHER


def _tab_cols():
    cols = [f"lc_l{l}" for l in range(1, 9)] + ["d1", "d2", "d4"]
    for c in WEATHER:
        cols += [f"{c}_l{l}" for l in range(1, 5)] + [f"{c}_r4"]
    return cols + ["sin_w", "cos_w"]


TAB_COLS = _tab_cols()


def add_base(df):
    d = df.sort_values("start_date").reset_index(drop=True).copy()
    d["lc"] = np.log1p(d["dengue_cases"])
    doy = d["start_date"].dt.dayofyear
    d["sin_w"] = np.sin(2 * np.pi * doy / 365.25)
    d["cos_w"] = np.cos(2 * np.pi * doy / 365.25)
    return d


def tabular(d):
    F = {}
    for l in range(1, 9):
        F[f"lc_l{l}"] = d["lc"].shift(l)
    F["d1"] = d["lc"].shift(1) - d["lc"].shift(2)
    F["d2"] = d["lc"].shift(2) - d["lc"].shift(3)
    F["d4"] = d["lc"].shift(1) - d["lc"].shift(5)
    for c in WEATHER:
        for l in range(1, 5):
            F[f"{c}_l{l}"] = d[c].shift(l)
        F[f"{c}_r4"] = d[c].shift(1).rolling(4).mean()
    F["sin_w"] = d["sin_w"]
    F["cos_w"] = d["cos_w"]
    return pd.DataFrame(F)[TAB_COLS]


def training_set(df):
    d = add_base(df)
    X = tabular(d)
    y = d["lc"] - d["lc"].shift(1)
    S = d[SEQ_COLS].values
    valid = np.where(X.notna().all(axis=1) & y.notna() & (np.arange(len(d)) >= SEQ))[0]
    return d, X, y, S, valid


def forecast_inputs(history, next_date=None):
    """Build model inputs for the week AFTER the last row of `history`.

    history: DataFrame with start_date, dengue_cases, temp_mean, rainfall_total, humidity_mean
             (at least SEQ rows; the last SEQ rows are used).
    Returns (tabular_row[1 x len(TAB_COLS)] DataFrame, seq[SEQ x len(SEQ_COLS)] ndarray, next_date, last_cases)
    """
    h = history[RAW_COLS].sort_values("start_date").tail(SEQ).copy()
    if len(h) < SEQ:
        raise ValueError(f"Need at least {SEQ} weeks of history, got {len(h)}.")
    if h[RAW_COLS].isna().any().any():
        raise ValueError("History contains missing values.")
    h["start_date"] = pd.to_datetime(h["start_date"])
    if next_date is None:
        next_date = h["start_date"].iloc[-1] + pd.Timedelta(days=7)
    nxt = pd.DataFrame({"start_date": [pd.Timestamp(next_date)], "dengue_cases": [np.nan],
                        **{c: [np.nan] for c in WEATHER}})
    d = add_base(pd.concat([h, nxt], ignore_index=True))
    X = tabular(d).iloc[[-1]]
    S = d[SEQ_COLS].values[:-1]
    return X, S, pd.Timestamp(next_date), float(h["dengue_cases"].iloc[-1])


LABELS = {
    "sin_w": "Season (week of year)", "cos_w": "Season (week of year)",
}


def friendly(name):
    if name in LABELS:
        return LABELS[name]
    if name.startswith("lc_l"):
        return f"Cases {name[4:]} wk ago"
    if name in ("d1", "d2", "d4"):
        return {"d1": "Case growth (last wk)", "d2": "Case growth (2 wk ago)", "d4": "Case growth (4-wk trend)"}[name]
    for c, lab in [("temp_mean", "Temperature"), ("rainfall_total", "Rainfall"), ("humidity_mean", "Humidity")]:
        if name.startswith(c):
            return f"{lab} (4-wk avg)" if name.endswith("_r4") else f"{lab}, {name[-1]} wk ago"
    return name
