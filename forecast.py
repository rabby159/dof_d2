"""Inference only: NumPy LSTM + XGBoost Booster. No TensorFlow needed to run the app.

The LSTM weights are exported to models/lstm_s{seed}.npz by train_pipeline.py
(Keras gate order: input, forget, cell, output).
"""
import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from features import TAB_COLS, forecast_inputs, friendly


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyLSTM:
    def __init__(self, path):
        z = np.load(path)
        self.W, self.U, self.b = z["kernel"], z["recurrent"], z["bias"]
        self.units = self.U.shape[0]

    def embed(self, seq):
        """seq: (T, n_features) -> last hidden state (units,)"""
        h = np.zeros(self.units)
        c = np.zeros(self.units)
        for x in seq:
            z = x @ self.W + h @ self.U + self.b
            i, f, g, o = np.split(z, 4)
            c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
            h = _sigmoid(o) * np.tanh(c)
        return h


class Forecaster:
    def __init__(self, model_dir="models"):
        with open(os.path.join(model_dir, "meta.json")) as f:
            self.meta = json.load(f)
        m = self.meta
        self.tab_mean, self.tab_scale = np.array(m["tab_mean"]), np.array(m["tab_scale"])
        self.seq_mean, self.seq_scale = np.array(m["seq_mean"]), np.array(m["seq_scale"])
        self.lstms, self.boosters = [], []
        for s in m["seeds"]:
            self.lstms.append(NumpyLSTM(os.path.join(model_dir, f"lstm_s{s}.npz")))
            b = xgb.Booster()
            b.load_model(os.path.join(model_dir, f"xgb_s{s}.json"))
            self.boosters.append(b)

    def predict(self, history, next_date=None):
        X, S, nd, last = forecast_inputs(history, next_date)
        Xs = ((X.values - self.tab_mean) / self.tab_scale)[0]
        Ss = (S - self.seq_mean) / self.seq_scale
        growths, tab_contrib, emb_contrib, bias = [], [], [], []
        for lstm, booster in zip(self.lstms, self.boosters):
            Z = np.hstack([Xs, lstm.embed(Ss)])[None, :]
            dm = xgb.DMatrix(Z)
            growths.append(float(booster.predict(dm)[0]))
            c = booster.predict(dm, pred_contribs=True)[0]
            tab_contrib.append(c[: len(TAB_COLS)])
            emb_contrib.append(c[len(TAB_COLS):-1].sum())
            bias.append(c[-1])
        g = float(np.mean(growths))
        lc_last = np.log1p(last)
        pred = float(np.expm1(lc_last + g))
        q10, q90 = self.meta["interval_log_q10"], self.meta["interval_log_q90"]
        low = float(np.expm1(np.log1p(pred) + q10))
        high = float(np.expm1(np.log1p(pred) + q90))
        contrib = pd.Series(np.mean(tab_contrib, axis=0), index=TAB_COLS)
        grouped = contrib.groupby(lambda n: friendly(n) if not n.startswith("lc_l") else "Recent case levels").sum()
        grouped["LSTM sequence embedding"] = float(np.mean(emb_contrib))
        return {
            "next_date": nd, "last_cases": last, "pred": max(pred, 0.0), "low": max(low, 0.0), "high": max(high, 0.0),
            "growth_log": g, "contrib": grouped.sort_values(key=np.abs, ascending=False),
            "base_growth": float(np.mean(bias)),
        }

    def risk_level(self, cases):
        t = self.meta["risk_thresholds"]
        if cases >= t["severe"]:
            return "SEVERE"
        if cases >= t["high"]:
            return "HIGH"
        if cases >= t["moderate"]:
            return "MODERATE"
        return "LOW"
