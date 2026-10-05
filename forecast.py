"""Inference: loads the saved hybrid LSTM-XGBoost ensemble and produces a forecast + explanation."""
import json
import os
import numpy as np
import pandas as pd
from features import TAB_COLS, forecast_inputs, friendly

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


class Forecaster:
    def __init__(self, model_dir="models"):
        import tensorflow as tf
        import xgboost as xgb
        with open(os.path.join(model_dir, "meta.json")) as f:
            self.meta = json.load(f)
        m = self.meta
        self.tab_mean, self.tab_scale = np.array(m["tab_mean"]), np.array(m["tab_scale"])
        self.seq_mean, self.seq_scale = np.array(m["seq_mean"]), np.array(m["seq_scale"])
        self.embedders, self.xgbs = [], []
        for s in m["seeds"]:
            lstm = tf.keras.models.load_model(os.path.join(model_dir, f"lstm_s{s}.keras"))
            self.embedders.append(tf.keras.models.Model(lstm.input, lstm.get_layer("emb").output))
            reg = xgb.XGBRegressor()
            reg.load_model(os.path.join(model_dir, f"xgb_s{s}.json"))
            self.xgbs.append(reg)

    def predict(self, history, next_date=None):
        import xgboost as xgb
        X, S, nd, last = forecast_inputs(history, next_date)
        Xs = (X.values - self.tab_mean) / self.tab_scale
        Ss = ((S - self.seq_mean) / self.seq_scale)[None, :, :].astype("float32")
        growths, tab_contrib, emb_contrib, bias = [], [], [], []
        for emb_model, reg in zip(self.embedders, self.xgbs):
            e = emb_model(Ss, training=False).numpy()
            Z = np.hstack([Xs, e])
            growths.append(float(reg.predict(Z)[0]))
            c = reg.get_booster().predict(xgb.DMatrix(Z), pred_contribs=True)[0]
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
