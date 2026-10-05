"""Train the final deployed model (3-seed hybrid LSTM-XGBoost ensemble) on ALL available weeks.

Honest performance numbers come from `evaluate.py` (walk-forward), not from this script.
Run order:  python evaluate.py  ->  python train_pipeline.py
"""
import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from xgboost import XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from features import SEQ, SEQ_COLS, TAB_COLS, training_set, friendly

CSV = "dengu_weather_dataset.csv"
SEEDS = [0, 1, 2]
EMB = 16
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv(CSV, parse_dates=["start_date"])
d, X, y, S, valid = training_set(df)
print(f"Rows: {len(d)} | trainable: {len(valid)} | last week: {d.start_date.iloc[-1].date()}")

tab_mean = X.iloc[valid].mean().values
tab_scale = X.iloc[valid].std(ddof=0).replace(0, 1).values
seq_mean, seq_scale = S.mean(axis=0), S.std(axis=0)
Xs = (X.iloc[valid].values - tab_mean) / tab_scale
Sn = (S - seq_mean) / seq_scale
A = np.stack([Sn[i - SEQ:i] for i in valid]).astype("float32")
yt = y.values[valid]


def make_xgb(seed):
    return XGBRegressor(n_estimators=250, learning_rate=0.03, max_depth=3, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=3, random_state=seed)


all_contrib, emb_names = [], [f"emb_{i}" for i in range(EMB)]
for s in SEEDS:
    tf.keras.utils.set_random_seed(s)
    inp = layers.Input(shape=(SEQ, len(SEQ_COLS)))
    h = layers.LSTM(EMB, name="emb")(inp)
    out = layers.Dense(1)(layers.Dropout(0.2)(h))
    lstm = Model(inp, out)
    lstm.compile(optimizer=tf.keras.optimizers.Adam(0.003), loss="mse")
    lstm.fit(A, yt, epochs=60, batch_size=16, verbose=0, validation_split=0.15,
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    emb = Model(lstm.input, lstm.get_layer("emb").output).predict(A, verbose=0)
    Z = np.hstack([Xs, emb])
    reg = make_xgb(s).fit(Z, yt)
    lstm.save(f"models/lstm_s{s}.keras")
    reg.save_model(f"models/xgb_s{s}.json")
    import xgboost as xgb
    c = reg.get_booster().predict(xgb.DMatrix(Z), pred_contribs=True)
    all_contrib.append(c)
    print(f"seed {s} trained")

# ---- Global TreeSHAP importance (XGBoost native TreeSHAP), averaged over seeds, embeddings grouped ----
C = np.mean([c[:, :len(TAB_COLS)] for c in all_contrib], axis=0)
Cemb = np.mean([c[:, len(TAB_COLS):-1].sum(axis=1) for c in all_contrib], axis=0)
imp = pd.Series(np.abs(C).mean(axis=0), index=TAB_COLS)
imp = imp.groupby(lambda n: "Recent case levels" if n.startswith("lc_l") else friendly(n)).sum()
imp["LSTM sequence embedding"] = np.abs(Cemb).mean()
imp = imp.sort_values()
plt.figure(figsize=(8, 5))
plt.barh(imp.index, imp.values, color="#2a5298")
plt.xlabel("Mean |TreeSHAP contribution| to weekly log-growth")
plt.title("Global feature importance (TreeSHAP, hybrid model)")
plt.tight_layout()
plt.savefig("outputs/shap_global.png", dpi=200)
plt.close()
imp.sort_values(ascending=False).to_csv("outputs/shap_global.csv")

# ---- Empirical 80% prediction interval from WALK-FORWARD residuals (log scale) ----
q10, q90 = -0.41, 0.36
if os.path.exists("outputs/predictions.csv"):
    p = pd.read_csv("outputs/predictions.csv")
    p = p[p.model == "Hybrid LSTM-XGBoost"]
    r = np.log1p(p.actual) - np.log1p(p.pred)
    q10, q90 = float(r.quantile(0.10)), float(r.quantile(0.90))
else:
    print("WARNING: outputs/predictions.csv missing - run evaluate.py first; using default interval.")

# ---- Percentile-based risk levels (weekly national cases since 2022, i.e. excluding 2020-21 under-reporting) ----
ref = df[df.start_date >= "2022-01-01"].dengue_cases
risk = {"moderate": float(ref.quantile(0.60)), "high": float(ref.quantile(0.75)), "severe": float(ref.quantile(0.90))}

meta = {
    "seeds": SEEDS, "seq": SEQ, "tab_cols": TAB_COLS, "seq_cols": SEQ_COLS,
    "tab_mean": [float(v) for v in tab_mean], "tab_scale": [float(v) for v in tab_scale],
    "seq_mean": [float(v) for v in seq_mean], "seq_scale": [float(v) for v in seq_scale],
    "interval_log_q10": q10, "interval_log_q90": q90,
    "risk_thresholds": risk,
    "data_range": {c: [float(df[c].min()), float(df[c].max())]
                   for c in ["dengue_cases", "temp_mean", "rainfall_total", "humidity_mean"]},
    "last_data_week": str(d.start_date.iloc[-1].date()), "first_data_week": str(d.start_date.iloc[0].date()),
    "n_weeks": int(len(d)),
}
with open("models/meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("Saved models/, outputs/shap_global.png, models/meta.json")
print("Risk thresholds:", {k: round(v) for k, v in risk.items()}, "| interval (log):", round(q10, 3), round(q90, 3))
