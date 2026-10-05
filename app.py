import os
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from features import RAW_COLS, SEQ
from forecast import Forecaster

DATA_CSV = "dengu_weather_dataset.csv"

st.set_page_config(page_title="Dengue Early Warning | GUB CSE Thesis", page_icon="🦟", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
 .main-header {background: linear-gradient(135deg,#1e3c72 0%,#2a5298 100%); padding:24px; border-radius:12px; color:white; margin-bottom:20px;}
 .main-header h1 {color:#fff; margin-bottom:4px; font-size:24px;}
 .main-header h3 {color:#e0e0e0; margin-top:0; font-size:15px; font-weight:400;}
 .meta-badge {background-color:rgba(255,255,255,.15); padding:6px 12px; border-radius:20px; font-size:12px; margin-right:8px; display:inline-block;}
 .risk {padding:20px; border-radius:10px; border:2px solid;}
 .risk-SEVERE {background:rgba(192,57,43,.20); border-color:#c0392b; color:#ff6b6b;}
 .risk-HIGH {background:rgba(231,76,60,.15); border-color:#e74c3c; color:#ff8a65;}
 .risk-MODERATE {background:rgba(241,196,15,.15); border-color:#f1c40f; color:#f39c12;}
 .risk-LOW {background:rgba(46,204,113,.15); border-color:#2ecc71; color:#2ecc71;}
</style>
<div class="main-header">
  <span class="meta-badge">🏛️ Green University of Bangladesh</span>
  <span class="meta-badge">💻 Dept. of Computer Science &amp; Engineering</span>
  <span class="meta-badge">🎓 Capstone Thesis Project</span>
  <h1>An Explainable AI (XAI) Enabled Hybrid LSTM-XGBoost Framework for Dengue Outbreak Forecasting</h1>
  <h3>Early Warning Decision Support System for Public Health Surveillance (academic prototype)</h3>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_forecaster():
    return Forecaster("models")


@st.cache_data
def load_data(path):
    return pd.read_csv(path, parse_dates=["start_date"]).sort_values("start_date").reset_index(drop=True)


try:
    fc_model = load_forecaster()
except Exception as e:  # noqa: BLE001
    st.error(f"Model artifacts could not be loaded. Run `python train_pipeline.py` first. Error: {e}")
    st.stop()

meta = fc_model.meta
base_df = load_data(DATA_CSV)

# ------------------------------------------------------------------ Sidebar
st.sidebar.title("🎛️ Forecast Input")
st.sidebar.caption(
    f"The model forecasts the week **after** the last row, using the previous {SEQ} weeks of cases and weather. "
    f"Bundled data ends with the week ending **{meta['last_data_week']}**.")
uploaded = st.sidebar.file_uploader("Optional: upload newer weekly data (CSV)", type="csv",
                                    help=f"Columns required: {', '.join(RAW_COLS)}")
data = base_df
if uploaded is not None:
    try:
        up = pd.read_csv(uploaded, parse_dates=["start_date"])
        missing = [c for c in RAW_COLS if c not in up.columns]
        if missing:
            st.sidebar.error(f"Missing columns: {missing}")
        else:
            data = pd.concat([base_df[RAW_COLS], up[RAW_COLS]]).drop_duplicates("start_date", keep="last") \
                     .sort_values("start_date").reset_index(drop=True)
            st.sidebar.success(f"Using data up to {data.start_date.iloc[-1].date()}")
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"Could not read file: {e}")

with st.sidebar.expander("👥 Research team & supervision"):
    st.markdown("**Supervisor:** Babe Sultana, Lecturer, Dept. of CSE, GUB  \n"
                "**Authors:** Md Rabby, Gazi Faria Akter, Md Julfikar Alam")

# ------------------------------------------------------------------ Input table (editable)
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Forecast & Early Warning", "📈 Trend & Inputs",
                                  "🔍 Explainability (TreeSHAP)", "🎓 Evaluation & Thesis"])

with tab2:
    st.subheader(f"Input window: last {SEQ} weeks (editable)")
    st.caption("Edit values if newer or corrected surveillance data is available. Values outside the training range "
               "trigger a warning because the model has not seen such conditions.")
    window = data[RAW_COLS].tail(SEQ).reset_index(drop=True)
    rng = meta["data_range"]
    edited = st.data_editor(
        window, num_rows="fixed", key="window_editor",
        column_config={
            "start_date": st.column_config.DateColumn("Week ending (Sun)", format="YYYY-MM-DD"),
            "dengue_cases": st.column_config.NumberColumn("Weekly cases", min_value=0, max_value=100000, step=1),
            "temp_mean": st.column_config.NumberColumn("Mean temp (°C)", min_value=5.0, max_value=45.0, step=0.1),
            "rainfall_total": st.column_config.NumberColumn("Rainfall (mm)", min_value=0.0, max_value=1000.0, step=0.1),
            "humidity_mean": st.column_config.NumberColumn("Mean humidity (%)", min_value=0.0, max_value=100.0, step=0.1),
        })

valid_input, problems = True, []
edited = edited.copy()
edited["start_date"] = pd.to_datetime(edited["start_date"])
if edited[RAW_COLS].isna().any().any():
    valid_input = False
    problems.append("The input table has empty cells.")
elif not edited["start_date"].is_monotonic_increasing:
    valid_input = False
    problems.append("Week-ending dates must be in increasing order.")

warnings_ood = []
if valid_input:
    for c, lab in [("dengue_cases", "cases"), ("temp_mean", "temperature"),
                   ("rainfall_total", "rainfall"), ("humidity_mean", "humidity")]:
        lo, hi = rng[c]
        if edited[c].min() < lo or edited[c].max() > hi:
            warnings_ood.append(f"{lab} is outside the training range ({lo:g} – {hi:g}).")

result = None
if valid_input:
    result = fc_model.predict(edited)

# ------------------------------------------------------------------ Tab 1
with tab1:
    if not valid_input:
        for p in problems:
            st.error(p)
    else:
        for w in warnings_ood:
            st.warning("⚠️ Out-of-range input: " + w + " Forecast reliability is reduced.")
        pred, low, high = result["pred"], result["low"], result["high"]
        level = fc_model.risk_level(pred)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"### 📊 Forecast for the week ending {result['next_date'].date()}")
            st.metric("Predicted weekly dengue cases", f"≈ {round(pred, -1):,.0f}",
                      delta=f"{pred - result['last_cases']:+,.0f} vs last week ({result['last_cases']:,.0f})",
                      delta_color="inverse")
            st.markdown(f"**80% prediction interval:** {round(low, -1):,.0f} – {round(high, -1):,.0f} cases")
            st.caption("Interval = empirical 10th–90th percentile of walk-forward forecast errors (see Evaluation tab). "
                       "Point forecast rounded to the nearest 10 to avoid false precision.")
        actions = {
            "LOW": ["Routine entomological and case surveillance.", "Keep monitoring weather and weekly case counts."],
            "MODERATE": ["Intensify larval source reduction (stagnant water, containers).",
                         "Community awareness and household container inspection."],
            "HIGH": ["Scale up targeted source-reduction campaigns in hotspots.",
                     "Prepare hospital capacity: dengue wards, triage, fluid-management supplies.",
                     "Issue public advisories on prevention and early care-seeking."],
            "SEVERE": ["Activate emergency coordination across health and city authorities.",
                       "Source reduction first; space spraying only as a supplementary outbreak response.",
                       "Expand hospital/ICU capacity and run media and citizen warning bulletins."],
        }
        icons = {"LOW": "✅", "MODERATE": "⚠️", "HIGH": "🚨", "SEVERE": "🆘"}
        t = meta["risk_thresholds"]
        with c2:
            st.markdown("### 🚨 Early Warning Level")
            lis = "".join(f"<li>{a}</li>" for a in actions[level])
            st.markdown(f"""<div class="risk risk-{level}"><h2>{icons[level]} {level} RISK</h2>
                <p>Forecast ≈ {round(pred, -1):,.0f} cases/week.</p><hr><b>Indicative actions:</b><ul>{lis}</ul></div>""",
                        unsafe_allow_html=True)
            st.caption(f"Levels are percentile-based on weekly cases since 2022: Moderate ≥ {t['moderate']:,.0f} (P60), "
                       f"High ≥ {t['high']:,.0f} (P75), Severe ≥ {t['severe']:,.0f} (P90). "
                       "Actions are indicative only and must be validated by public-health authorities.")

# ------------------------------------------------------------------ Tab 2 (chart + heuristic index)
with tab2:
    if valid_input:
        st.markdown("---")
        st.subheader("Recent trend and forecast")
        hist = pd.concat([data[["start_date", "dengue_cases"]], edited[["start_date", "dengue_cases"]]]) \
            .drop_duplicates("start_date", keep="last").sort_values("start_date").tail(30)
        fc_df = pd.DataFrame({"start_date": [result["next_date"]], "pred": [result["pred"]],
                              "low": [result["low"]], "high": [result["high"]]})
        line = alt.Chart(hist).mark_line(point=True).encode(
            x=alt.X("start_date:T", title="Week"), y=alt.Y("dengue_cases:Q", title="Weekly cases"))
        band = alt.Chart(fc_df).mark_rule(size=4, color="#e74c3c").encode(x="start_date:T", y="low:Q", y2="high:Q")
        pt = alt.Chart(fc_df).mark_point(size=140, filled=True, color="#e74c3c").encode(x="start_date:T", y="pred:Q")
        st.altair_chart((line + band + pt).properties(width="container", height=320))
        st.caption("Blue: reported weekly cases. Red: forecast with 80% interval.")

        st.markdown("---")
        st.subheader("🦟 Breeding suitability indicator (rule-based, not a model output)")
        last = edited.iloc[-1]
        temp_score = 100 if 25 <= last.temp_mean <= 30 else (50 if 20 <= last.temp_mean <= 35 else 20)
        rain_score = 100 if last.rainfall_total >= 50 else last.rainfall_total * 2
        hum_score = 100 if last.humidity_mean >= 75 else last.humidity_mean
        idx = int(round((temp_score + rain_score + hum_score) / 3))
        st.progress(idx / 100)
        st.caption(f"Indicator for the latest input week: **{idx}%**. Heuristic scoring: 25–30 °C, rainfall ≥ 50 mm, "
                   "humidity ≥ 75 % score highest. It does not feed the forecast; cite a published source if used in the thesis.")

# ------------------------------------------------------------------ Tab 3
with tab3:
    st.subheader("Why this forecast? (TreeSHAP on the XGBoost stage)")
    if valid_input:
        contrib = result["contrib"]
        cdf = pd.DataFrame({"feature": contrib.index, "effect": contrib.values * 100})
        cdf = cdf.reindex(cdf.effect.abs().sort_values(ascending=False).index).head(10)
        chart = alt.Chart(cdf).mark_bar().encode(
            x=alt.X("effect:Q", title="Effect on forecast weekly growth (percentage points, log scale)"),
            y=alt.Y("feature:N", sort=None, title=None),
            color=alt.condition(alt.datum.effect > 0, alt.value("#e74c3c"), alt.value("#2ecc71")))
        st.altair_chart(chart.properties(width="container", height=320))
        st.caption(f"Predicted weekly growth: {result['growth_log'] * 100:+.1f}% (log scale). Red bars push the forecast up, "
                   "green bars push it down, relative to the model's average growth. All 16 LSTM embedding dimensions are "
                   "summed into one group because individual dimensions are not human-interpretable.")
    st.markdown("---")
    st.subheader("Global feature importance")
    if os.path.exists("outputs/shap_global.png"):
        st.image("outputs/shap_global.png")
    st.info("In this dataset, recent case dynamics and seasonality dominate the forecast; weather variables contribute "
            "comparatively little. This is reported as a finding, not hidden.")

# ------------------------------------------------------------------ Tab 4
with tab4:
    st.subheader("Evaluation protocol")
    st.markdown(f"""
* **Task:** true 1-week-ahead forecast. Inputs use only information up to the previous week (past cases, past weather) and the
  calendar week of the target. Same-week weather is **not** used.
* **Data:** {meta['n_weeks']} weekly records, {meta['first_data_week']} → {meta['last_data_week']} (national weekly dengue cases and
  weather; weeks are Mon–Sun, labelled by the Sunday). 2020–2021 contain near-zero reported cases (suspected under-reporting) and are kept but flagged as a limitation.
* **Validation:** expanding-window walk-forward, test years 2023, 2024, 2025, 2026 (to September); scalers and models fitted only on past data.
* **Metric:** MASE relative to the "next week = last week" baseline (**MASE < 1 beats persistence**), plus MAE, RMSE and MAPE.
""")
    if os.path.exists("outputs/evaluation_table.csv"):
        ev = pd.read_csv("outputs/evaluation_table.csv").set_index("model")
        st.dataframe(ev.round(3))
        st.caption("3 random seeds per neural model; results from `evaluate.py`. LSTM-only and the hybrid perform similarly; "
                   "the hybrid is clearly better than XGBoost-only, which fails on the unprecedented 2023 outbreak.")
    st.subheader("Limitations")
    st.markdown("""
* One country-level series with a single exceptional outbreak (2023); results may not generalise to other regions or outbreak sizes.
* Short history (7 seasons) and COVID-era reporting gaps.
* Prediction intervals are empirical and wide (roughly −34% to +43% around the point forecast).
* Weather adds little predictive information beyond past cases and seasonality in this dataset.
* Research prototype, not an operational public-health tool.
""")
