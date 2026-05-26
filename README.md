# Time Series Forecasting: Energy Consumption Prediction

A complete data science project demonstrating how to forecast future energy consumption using three complementary approaches: classical statistical modeling (ARIMA), modern decomposition-based forecasting (Prophet), and deep learning (LSTM). The project walks through the full pipeline from raw data generation and exploratory analysis through model training, evaluation, and comparison.

---

## Project Overview

Energy consumption data exhibits rich temporal structure — long-term growth trends, weekly usage cycles, and yearly seasonal swings tied to heating and cooling demand. This project uses three years of synthetic daily energy readings (in kWh) to showcase how each modeling family handles these patterns.

| Model | Family | Strengths |
|---|---|---|
| ARIMA | Classical statistics | Interpretable, principled stationarity handling |
| Prophet | Decomposition / Bayesian | Automatic seasonality, robust to missing data |
| LSTM | Deep learning (Keras) | Learns non-linear long-range dependencies |

---

## Project Structure

```
time-series-forecasting/
├── time_series_forecasting.ipynb   # Main analysis notebook (10 cells)
├── requirements.txt                # Pinned Python dependencies
├── README.md                       # This file
└── data/                           # Placeholder for any external datasets
```

---

## Setup Instructions

### 1. Clone / navigate to the project

```bash
cd /path/to/time-series-forecasting
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow 2.15 requires Python 3.8–3.11. On Apple Silicon Macs, use `tensorflow-macos` and `tensorflow-metal` instead of `tensorflow==2.15.0`.

### 4. Launch Jupyter

```bash
jupyter notebook time_series_forecasting.ipynb
```

Run all cells in order (Cell > Run All) or step through them individually.

---

## Notebook Walkthrough

### Cell 1 — Introduction
Markdown overview of the problem statement, dataset, and the three models used.

### Cell 2 — Imports
All required libraries: pandas, numpy, matplotlib, statsmodels, Prophet, TensorFlow/Keras, and scikit-learn.

### Cell 3 — Synthetic Data Generation
Three years of daily readings (2020-01-01 to 2022-12-31) constructed from four additive components:
- **Trend**: slow linear growth (+0.05 kWh/day)
- **Weekly seasonality**: lower consumption on weekends
- **Yearly seasonality**: peak in winter, trough in summer (cosine wave)
- **Gaussian noise**: random day-to-day variation

Output: a `pandas.DataFrame` with columns `date` and `consumption_kwh`.

### Cell 4 — Exploratory Data Analysis
- Raw time series plot
- 30-day and 90-day rolling mean and standard deviation overlaid
- Classical seasonal decomposition (additive model, period = 365)

### Cell 5 — Stationarity Testing and Differencing
- Augmented Dickey-Fuller (ADF) test on the original series
- First-order differencing to achieve stationarity
- ADF re-test to confirm

### Cell 6 — ARIMA Model
- ACF and PACF plots used to identify lag structure
- ARIMA(2,1,2) fitted via `statsmodels.tsa.arima.model.ARIMA`
- 60-day out-of-sample forecast on a held-out test set
- Evaluation: RMSE and MAE

### Cell 7 — Prophet Model
- Data reformatted to Prophet's `ds`/`y` convention
- Model fitted with yearly and weekly seasonality enabled
- 60-day forecast with uncertainty intervals
- Component plots (trend, weekly effect, yearly effect)

### Cell 8 — LSTM Model
- Min-max scaling of the training series
- Sliding-window sequence creation (look-back = 30 days)
- Architecture: `Input → LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1)`
- Trained for 30 epochs (batch size 32)
- Predictions inverse-transformed and evaluated

### Cell 9 — Model Comparison
- All three forecast series plotted against the actual test values on a single chart
- Summary metrics table (RMSE, MAE) for each model

### Cell 10 — Conclusions
Markdown discussion of results, model trade-offs, and suggested next steps.

---

## Methodology

### Why ARIMA?
ARIMA (AutoRegressive Integrated Moving Average) is the workhorse of classical time series analysis. It models the series as a linear combination of its own past values and past forecast errors. The "Integrated" component handles non-stationarity through differencing. ARIMA is highly interpretable and works well when the underlying dynamics are approximately linear.

### Why Prophet?
Developed at Meta, Prophet decomposes a series into trend, seasonality, and holiday components using a curve-fitting approach. It requires minimal hyperparameter tuning, handles missing data gracefully, and produces well-calibrated uncertainty intervals. Its component plots make seasonal patterns easy to communicate to non-technical stakeholders.

### Why LSTM?
Long Short-Term Memory networks are recurrent neural networks specifically designed to learn long-range temporal dependencies. Unlike ARIMA and Prophet, LSTM can capture non-linear interactions between variables and adapts to structural breaks in the series. The trade-off is greater data requirements and reduced interpretability.

---

## Key Results (indicative — values vary by random seed)

| Model | RMSE (kWh) | MAE (kWh) |
|---|---|---|
| ARIMA | ~15–25 | ~12–20 |
| Prophet | ~10–18 | ~8–15 |
| LSTM | ~12–22 | ~9–18 |

Exact numbers depend on the random seed used during data generation and LSTM weight initialization.

---

## References

- Box, G. E. P., Jenkins, G. M., et al. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.)
- Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37-45.
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.
- Hands-On Time Series Forecasting — [Medium / Data Science Collective](https://medium.com/data-science-collective/hands-on-time-series-forecasting-43ccbd418c9a)
