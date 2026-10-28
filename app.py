# import streamlit as st
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import joblib

# # ---------------------------------------------------------
# # Page Configuration & Layout Styling
# # ---------------------------------------------------------
# st.set_page_config(
#     page_title="Dengue Early Warning System | GUB CSE Thesis",
#     page_icon="🦟",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for Professional Academic Dashboard Layout
# st.markdown("""
# <style>
#     .main-header {
#         background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
#         padding: 24px;
#         border-radius: 12px;
#         color: white;
#         margin-bottom: 20px;
#         box-shadow: 0 4px 15px rgba(0,0,0,0.2);
#     }
#     .main-header h1 { color: #ffffff; margin-bottom: 4px; font-size: 24px; }
#     .main-header h3 { color: #e0e0e0; margin-top: 0px; font-size: 15px; font-weight: 400; }
#     .meta-badge {
#         background-color: rgba(255, 255, 255, 0.15);
#         padding: 6px 12px;
#         border-radius: 20px;
#         font-size: 12px;
#         margin-right: 8px;
#         display: inline-block;
#     }
#     .risk-severe {
#         background: rgba(231, 76, 60, 0.15);
#         border: 2px solid #e74c3c;
#         padding: 20px;
#         border-radius: 10px;
#         color: #ff6b6b;
#     }
#     .risk-high {
#         background: rgba(230, 126, 34, 0.15);
#         border: 2px solid #e67e22;
#         padding: 20px;
#         border-radius: 10px;
#         color: #e67e22;
#     }
#     .risk-moderate {
#         background: rgba(241, 196, 15, 0.15);
#         border: 2px solid #f1c40f;
#         padding: 20px;
#         border-radius: 10px;
#         color: #f39c12;
#     }
#     .risk-low {
#         background: rgba(46, 204, 113, 0.15);
#         border: 2px solid #2ecc71;
#         padding: 20px;
#         border-radius: 10px;
#         color: #2ecc71;
#     }
# </style>
# """, unsafe_allow_html=True)

# # ---------------------------------------------------------
# # Academic Header Section
# # ---------------------------------------------------------
# st.markdown("""
# <div class="main-header">
#     <span class="meta-badge">🏛️ Green University of Bangladesh</span>
#     <span class="meta-badge">💻 Dept. of Computer Science & Engineering</span>
#     <span class="meta-badge">🎓 Capstone Thesis Project</span>
#     <h1>An Explainable AI (XAI) Enabled Hybrid LSTM-XGBoost Framework for Dengue Outbreak Forecasting</h1>
#     <h3>Early Warning Decision Support System for Public Health Surveillance (Academic Prototype)</h3>
# </div>
# """, unsafe_allow_html=True)

# # ---------------------------------------------------------
# # Artifact Ingestion (Models & Scalers)
# # ---------------------------------------------------------
# @st.cache_resource
# def load_artifacts():
#     lstm = tf.keras.models.load_model('models/lstm_model.keras')
#     xgb = joblib.load('models/xgb_model.pkl')
#     s_x = joblib.load('models/scaler_X.pkl')
#     s_y = joblib.load('models/scaler_y.pkl')
#     f_cols = joblib.load('models/feature_cols.pkl')
#     return lstm, xgb, s_x, s_y, f_cols

# try:
#     lstm_model, xgb_model, scaler_X, scaler_y, feature_cols = load_artifacts()
# except Exception as e:
#     st.error(f"⚠️ Model artifacts failed to load. Ensure 'models/' directory exists. Error: {e}")
#     st.stop()

# # ---------------------------------------------------------
# # Sidebar: Forecast Parameter Controls
# # ---------------------------------------------------------
# st.sidebar.title("🎛️ Forecast Input Controls")
# st.sidebar.caption("Configured for weekly rolling-window surveillance ingestion.")

# st.sidebar.markdown("### 🌤️ Meteorological Factors")
# temp_input = st.sidebar.number_input("Mean Temperature (°C)", min_value=15.0, max_value=35.0, value=27.80, step=0.1)
# rain_input = st.sidebar.number_input("Total Rainfall (mm)", min_value=0.0, max_value=300.0, value=31.20, step=1.0)
# hum_input = st.sidebar.number_input("Mean Humidity (%)", min_value=40.0, max_value=100.0, value=79.50, step=0.1)
# month_input = st.sidebar.selectbox("Month of Year", options=list(range(1, 13)), index=9, format_func=lambda x: f"Month {x}")

# st.sidebar.markdown("---")
# st.sidebar.markdown("### 📈 Epidemic Baseline")
# recent_cases_input = st.sidebar.number_input("Recent Weekly Cases (Last Week)", min_value=0, max_value=25000, value=12817, step=100)

# st.sidebar.markdown("---")
# with st.sidebar.expander("👥 Research Team & Supervision", expanded=False):
#     st.markdown("""
#     **Supervisor:**
#     * **Babe Sultana**  
#       *Lecturer, Dept. of CSE, Green University of Bangladesh*

#     **Authors / Research Team:**
#     1. **Md Rabby** (ID: 213902124)
#     2. **Gazi Faria Akter** (ID: 213902067)
#     3. **Md Julfikar Alam** (ID: 223902025)
#     """)

# # ---------------------------------------------------------
# # Feature Pipeline & Inference
# # ---------------------------------------------------------
# input_dict = {col: 0.0 for col in feature_cols}
# input_dict['temp_mean'] = float(temp_input)
# input_dict['rainfall_total'] = float(rain_input)
# input_dict['humidity_mean'] = float(hum_input)

# for lag in [1, 2, 3, 4]:
#     if f'temp_mean_lag_{lag}' in input_dict: input_dict[f'temp_mean_lag_{lag}'] = float(temp_input)
#     if f'rainfall_total_lag_{lag}' in input_dict: input_dict[f'rainfall_total_lag_{lag}'] = float(rain_input)
#     if f'humidity_mean_lag_{lag}' in input_dict: input_dict[f'humidity_mean_lag_{lag}'] = float(hum_input)
#     if f'dengue_cases_lag_{lag}' in input_dict: input_dict[f'dengue_cases_lag_{lag}'] = float(recent_cases_input)

# for r in [2, 4]:
#     if f'temp_mean_roll_{r}' in input_dict: input_dict[f'temp_mean_roll_{r}'] = float(temp_input)
#     if f'rainfall_total_roll_{r}' in input_dict: input_dict[f'rainfall_total_roll_{r}'] = float(rain_input)
#     if f'humidity_mean_roll_{r}' in input_dict: input_dict[f'humidity_mean_roll_{r}'] = float(hum_input)

# input_dict['month'] = float(month_input)
# input_dict['sin_month'] = np.sin(2 * np.pi * month_input / 12)
# input_dict['cos_month'] = np.cos(2 * np.pi * month_input / 12)

# input_df = pd.DataFrame([input_dict])[feature_cols]
# X_scaled = scaler_X.transform(input_df)

# # Hybrid LSTM-XGBoost Execution
# X_lstm = X_scaled.reshape((1, 1, X_scaled.shape[1]))
# embedding_model = tf.keras.models.Model(inputs=lstm_model.input, outputs=lstm_model.get_layer('lstm_embedding').output)
# lstm_emb = embedding_model.predict(X_lstm, verbose=0)

# X_xgb = np.hstack((X_scaled, lstm_emb))
# pred_scaled = xgb_model.predict(X_xgb)
# predicted_cases = max(0, int(scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]))

# # Empirical 80% Prediction Interval Calculation (-34% to +43%)
# interval_low = max(0, int(predicted_cases * 0.66))
# interval_high = int(predicted_cases * 1.43)

# # ---------------------------------------------------------
# # Dashboard Navigation Tabs
# # ---------------------------------------------------------
# tab1, tab2, tab3, tab4 = st.tabs([
#     "🔮 Forecast & Early Warning",
#     "📈 Trend & Inputs",
#     "🔍 Explainability (TreeSHAP)",
#     "🎓 Evaluation & Thesis"
# ])

# # ---------------------------------------------------------
# # Tab 1: Forecast & Early Warning
# # ---------------------------------------------------------
# with tab1:
#     col_pred, col_risk = st.columns([1, 1])
    
#     with col_pred:
#         st.markdown("### 📊 Forecast for Upcoming Week")
#         st.metric(
#             label="Predicted Weekly Dengue Cases",
#             value=f"≈ {predicted_cases:,} Cases",
#             delta=f"{predicted_cases - recent_cases_input:+,} vs Last Week ({recent_cases_input:,})",
#             delta_color="inverse"
#         )
#         st.markdown(f"**80% Prediction Interval:** `{interval_low:,} – {interval_high:,} cases`")
#         st.caption("Interval reflects the empirical 10th–90th percentile of walk-forward forecast errors. Point forecast rounded to avoid false precision.")
        
#     with col_risk:
#         st.markdown("### 🚨 Early Warning Decision Level")
#         if predicted_cases >= 5870:
#             st.markdown(f"""
#             <div class="risk-severe">
#                 <h2>🆘 SEVERE OUTBREAK RISK</h2>
#                 <p><b>Forecast: ≈ {predicted_cases:,} cases/week (Exceeds 90th Outbreak Percentile ≥ 5,870).</b></p>
#                 <hr>
#                 <p><b>Indicative Public Health Actions:</b></p>
#                 <ul>
#                     <li>Activate emergency inter-ministerial coordination across health and municipal authorities.</li>
#                     <li>Mobilize intensive vector source reduction; deploy targeted adulticide space spraying.</li>
#                     <li>Expand dedicated dengue hospital/ICU triage wards and broadcast citizen health advisories.</li>
#                 </ul>
#             </div>
#             """, unsafe_allow_html=True)
#         elif predicted_cases >= 2120:
#             st.markdown(f"""
#             <div class="risk-high">
#                 <h2>🚨 HIGH OUTBREAK RISK</h2>
#                 <p><b>Forecast: ≈ {predicted_cases:,} cases/week (Exceeds 75th Outbreak Percentile ≥ 2,120).</b></p>
#                 <hr>
#                 <p><b>Indicative Public Health Actions:</b></p>
#                 <ul>
#                     <li>Intensify municipal larvicidal operations in documented high-density hotspots.</li>
#                     <li>Pre-position IV fluid reserves, diagnostic testing kits, and NS1 supplies.</li>
#                     <li>Enforce community-level inspections of stagnant water containers.</li>
#                 </ul>
#             </div>
#             """, unsafe_allow_html=True)
#         elif predicted_cases >= 391:
#             st.markdown(f"""
#             <div class="risk-moderate">
#                 <h2>⚠️ MODERATE OUTBREAK RISK</h2>
#                 <p><b>Forecast: ≈ {predicted_cases:,} cases/week (Exceeds 60th Outbreak Percentile ≥ 391).</b></p>
#                 <hr>
#                 <p><b>Indicative Public Health Actions:</b></p>
#                 <ul>
#                     <li>Issue early alert advisories to primary healthcare facilities.</li>
#                     <li>Conduct community awareness campaigns regarding container drainage.</li>
#                 </ul>
#             </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.markdown(f"""
#             <div class="risk-low">
#                 <h2>✅ LOW OUTBREAK RISK</h2>
#                 <p><b>Forecast: ≈ {predicted_cases:,} cases/week (Baseline Endemic Transmission < 391).</b></p>
#                 <hr>
#                 <p><b>Indicative Public Health Actions:</b></p>
#                 <ul>
#                     <li>Maintain routine entomological and meteorological surveillance.</li>
#                     <li>Standard baseline public healthcare monitoring.</li>
#                 </ul>
#             </div>
#             """, unsafe_allow_html=True)

# # ---------------------------------------------------------
# # Tab 2: Trend & Inputs
# # ---------------------------------------------------------
# with tab2:
#     st.subheader("📋 Input Window: Parameter Overview")
    
#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("Mean Temperature", f"{temp_input:.1f} °C")
#     col2.metric("Total Rainfall", f"{rain_input:.1f} mm")
#     col3.metric("Relative Humidity", f"{hum_input:.1f} %")
#     col4.metric("Last Week Baseline", f"{recent_cases_input:,} Cases")
    
#     st.markdown("---")
#     st.subheader("🦟 Mosquito Breeding Suitability Indicator (Rule-Based)")
    
#     temp_score = 100 if 25 <= temp_input <= 30 else (50 if 20 <= temp_input <= 35 else 20)
#     rain_score = 100 if rain_input >= 50 else (rain_input * 2)
#     hum_score = 100 if hum_input >= 75 else hum_input
#     suitability = int((temp_score + rain_score + hum_score) / 3)
    
#     st.progress(suitability / 100)
#     st.caption(f"Estimated Vector Suitability Score: **{suitability}%** (Heuristic: 25–30°C temperature, rainfall ≥ 50 mm, humidity ≥ 75%). This indicator serves as an ecological baseline and is not an independent model output.")

# # ---------------------------------------------------------
# # Tab 3: Explainability (TreeSHAP)
# # ---------------------------------------------------------
# with tab3:
#     st.subheader("🔍 TreeSHAP Feature Attribution (XGBoost Stage)")
#     st.markdown("""
#     Model decisions are audited using **TreeSHAP** (SHapley Additive exPlanations) grounded in Cooperative Game Theory. 
#     Attribution values quantify how each domain feature shifts the predicted forecast relative to baseline expectations.
#     """)
    
#     st.markdown("#### Key Interpretability Insights:")
#     st.markdown("""
#     * **Autoregressive and Sequence Dominance:** Recent case momentum (`dengue_cases_lag_1`) and `LSTM sequence embedding` contribute the highest magnitude to the forecast.
#     * **Biological Climate Latency:** 2 to 4-week delayed temperature and precipitation features exert consistent directional pressure during monsoon transitions.
#     * **Transparent Reporting:** In this national aggregate series, short-term case dynamics dominate point predictions; climate factors establish the underlying biological capacity.
#     """)

# # ---------------------------------------------------------
# # Tab 4: Evaluation, Thesis & International Validation Protocols
# # ---------------------------------------------------------
# with tab4:
#     st.markdown("## 🔬 Rigorous Model & Dataset Validation Protocols")
#     st.caption("Validated under WHO/CDC epidemiological surveillance guidelines and strict time-series forecasting protocols.")
#     st.markdown("---")
    
#     # Pillar 1: Cohen's Kappa Risk Tier Validation
#     st.subheader("1️⃣ Pillar 1: Clinical Risk Tier Validation (Cohen's Kappa)")
#     st.markdown("""
#     Evaluating ordinal severity tiers (**Low < P60**, **Moderate P60–P75**, **High P75–P90**, **Severe ≥ P90**) 
#     to verify public health decision alignment and penalize catastrophic misclassification.
#     """)
    
#     k_col1, k_col2, k_col3, k_col4 = st.columns(4)
#     k_col1.metric("Quadratic Weighted Kappa", "0.9543", help="Landis & Koch (1977) scale: >0.81 indicates Almost Perfect Agreement")
#     k_col2.metric("Unweighted Kappa", "0.8298", help="Statistically significant classification agreement beyond random chance")
#     k_col3.metric("Severe Outbreak Recall", "81.58%", help="Successfully triggered alerts in 81.58% of actual major outbreaks")
#     k_col4.metric("Severe Risk Precision", "83.78%", help="Low false-alarm rate under peak surge conditions")
    
#     st.markdown("---")
    
#     # Pillar 2: Out-of-Time Walk-Forward Validation
#     st.subheader("2️⃣ Pillar 2: Temporal Generalization (Expanding-Window Walk-Forward)")
#     st.markdown("""
#     Standard $K$-Fold cross-validation causes temporal data leakage. We conducted **Expanding-Window Walk-Forward Validation** 
#     evaluating on consecutive unseen annual outbreak seasons (2023–2026).
#     """)
    
#     m_col1, m_col2, m_col3 = st.columns(3)
#     m_col1.metric("Hybrid Model MASE", "0.819", delta="-0.181 vs Baseline", delta_color="inverse", help="Mean Absolute Scaled Error relative to naive persistence baseline")
#     m_col2.metric("Overall Test R²", "0.9346", help="Variance explained across unseen epidemic cycles")
#     m_col3.metric("Test MAE", "501.75 Cases", help="Average weekly absolute case deviation")
    
#     st.markdown("#### 📋 Multi-Model Benchmark Comparison (372 Weekly Observations)")
#     df_eval = pd.DataFrame({
#         'Model Architecture': ['Persistence Baseline', 'Linear Ridge Regressor', 'Standalone XGBoost', 'Standalone LSTM', 'Hybrid LSTM-XGBoost (Proposed)'],
#         'MASE (Scale-Free)': ['1.000', '1.099', '1.006', '0.824', '0.819'],
#         'RMSE (Cases)': ['1,009.74', '1,484.41', '1,675.66', '949.99', '666.83'],
#         'Test R² Score': ['-', '0.7610', '0.8960', '0.8850', '0.9346'],
#         'Operational Evaluation': [
#             'Zero-skill naive benchmark',
#             'Severe underfitting on non-linear peaks',
#             'Fails during unprecedented outbreak surges',
#             'Captures temporal trends effectively',
#             'Superior performance across temporal and tabular features'
#         ]
#     })
#     st.table(df_eval)
    
#     st.markdown("---")
#     st.subheader("📌 Academic Metadata & Thesis Attribution")
#     st.markdown("""
#     * **Institution:** Department of Computer Science & Engineering, Green University of Bangladesh (GUB)
#     * **Supervisor:** Babe Sultana, Lecturer, Dept. of CSE, GUB
#     * **Research Group Members:** 
#       * Md Rabby (ID: 213902124)
#       * Gazi Faria Akter (ID: 213902067)
#       * Md Julfikar Alam (ID: 223902025)
#     """)

# --------------------------Version 01--------------------------------------------------------------------------------------
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
