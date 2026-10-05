# 🦟 Explainable Hybrid LSTM-XGBoost Framework for Dengue Outbreak Forecasting

Capstone thesis, Dept. of CSE, Green University of Bangladesh.
Supervisor: Babe Sultana. Authors: Md Rabby, Gazi Faria Akter, Md Julfikar Alam.

Live app: https://dengue26.streamlit.app/

## What it does
Forecasts national weekly dengue cases in Bangladesh **one week ahead** and maps the forecast to an early-warning level
(Low / Moderate / High / Severe). The forecast comes with an empirical 80% prediction interval and a TreeSHAP explanation.

**Task definition.** For week *t*, the model uses only information up to week *t-1* (8 weeks of past cases and weather) plus the
calendar position of week *t* (weeks are labelled by their Sunday). Same-week weather is not used. Target: weekly log growth of cases.

**Model.** An LSTM reads the last 8 weeks (cases, temperature, rainfall, humidity, seasonality) and produces a 16-dimensional
embedding. The embedding is concatenated with tabular lag/rolling features and fed to an XGBoost regressor. The deployed model is a
3-seed ensemble. TreeSHAP (XGBoost native `pred_contribs`) explains the XGBoost stage; the 16 embedding dimensions are grouped.

## Results (expanding-window walk-forward, test years 2023-2026, 196 weeks)
| Model | MAE | RMSE | MASE | MAPE % |
|---|---|---|---|---|
| Persistence | 539 | 1010 | 1.00 | 27.5 |
| Ridge | 593 | 1484 | 1.10 | 23.0 |
| XGBoost-only | 542 | 1676 | 1.01 | 23.0 |
| LSTM-only | 444 | 950 | 0.82 | 22.7 |
| Hybrid LSTM-XGBoost | 442 | 1103 | 0.82 | 22.7 |

MASE < 1 means better than "next week = last week". Hybrid and LSTM-only are similar; the hybrid is clearly better than
XGBoost-only, which fails on the unprecedented 2023 outbreak. Weather contributes little beyond past cases and seasonality
(see global TreeSHAP plot in `outputs/shap_global.png`). Reproduce with `python evaluate.py`.

## Run
```bash
pip install -r requirements.txt          # app only (no TensorFlow needed)
streamlit run app.py

pip install -r requirements-train.txt    # only to retrain / re-evaluate (TensorFlow)
python evaluate.py          # walk-forward evaluation -> outputs/
python train_pipeline.py    # final model -> models/, outputs/shap_global.png

pip install -r requirements-dev.txt
python -m pytest -q
```

## Data
Weekly national dengue cases and weather, 2019-08-21 to 2026-09-27 (372 weeks), file `dengu_weather_dataset.csv`.
Each row is one Mon-Sun week, **labelled by its last day (Sunday)**; the column is named `start_date` for historical reasons.
Cases: DGHS dengue dashboard, "Dengue affected (Admitted) by date", summed over the 7 days of each week
(spot-check: 21-27 Sep 2026 = 11,722, matching the dataset). Hospital admissions, not all infections.
**TODO (authors): add the dashboard URL, access date and the weather data source.**
2020-2021 contain near-zero reported cases and are kept as a documented limitation.

## Prospective check (logged before the data were available)
| Forecast made | Target week (Mon-Sun) | Forecast | Actual (DGHS) | Abs. error | Persistence error |
|---|---|---|---|---|---|
| 2026-10-04 | 28 Sep - 4 Oct 2026 | 12,750 | 12,764 | 0.1% | 8.2% (11,722) |

A single point is a case study, not evidence of accuracy (walk-forward MAPE is about 22%). Keep logging weekly.

## Limitations
Single country-level series with one exceptional outbreak (2023); short history; wide prediction intervals; weather adds little
information in this dataset. Early-warning levels are percentile-based (P60/P75/P90 of weekly cases since 2022), not clinically
validated. Academic prototype, not an operational public-health tool.

## License
MIT
