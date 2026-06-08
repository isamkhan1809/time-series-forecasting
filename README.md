<div align="center">

```
████████╗██╗███╗   ███╗███████╗    ███████╗███████╗██████╗ ██╗███████╗███████╗
╚══██╔══╝██║████╗ ████║██╔════╝    ██╔════╝██╔════╝██╔══██╗██║██╔════╝██╔════╝
   ██║   ██║██╔████╔██║█████╗      ███████╗█████╗  ██████╔╝██║█████╗  ███████╗
   ██║   ██║██║╚██╔╝██║██╔══╝      ╚════██║██╔══╝  ██╔══██╗██║██╔══╝  ╚════██║
   ██║   ██║██║ ╚═╝ ██║███████╗    ███████║███████╗██║  ██║██║███████╗███████║
   ╚═╝   ╚═╝╚═╝     ╚═╝╚══════╝    ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝

    ███████╗ ██████╗ ██████╗ ███████╗ ██████╗ █████╗ ███████╗████████╗██╗███╗   ██╗ ██████╗
    ██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝╚══██╔══╝██║████╗  ██║██╔════╝
    █████╗  ██║   ██║██████╔╝█████╗  ██║     ███████║███████╗   ██║   ██║██╔██╗ ██║██║  ███╗
    ██╔══╝  ██║   ██║██╔══██╗██╔══╝  ██║     ██╔══██║╚════██║   ██║   ██║██║╚██╗██║██║   ██║
    ██║     ╚██████╔╝██║  ██║███████╗╚██████╗██║  ██║███████║   ██║   ██║██║ ╚████║╚██████╔╝
    ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝
```

### *ARIMA. Prophet. LSTM. Three Models. One Future.*

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Prophet](https://img.shields.io/badge/Prophet-Meta-4267B2?style=for-the-badge)](https://facebook.github.io/prophet)
[![statsmodels](https://img.shields.io/badge/ARIMA-statsmodels-blue?style=for-the-badge)](https://www.statsmodels.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

> **A complete time series forecasting project that trains ARIMA, Prophet, and LSTM on 3 years of energy consumption data — then pits them head-to-head on a 60-day forecast.**

</div>

---

## ◈ Three Paradigms. One Problem.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    FORECASTING ARCHITECTURE                          │
│                                                                      │
│   3 Years Daily Energy Data (kWh)                                    │
│   Trend + Weekly + Yearly seasonality + Noise                        │
│              │                                                       │
│     ┌────────┴──────────────────────────┐                           │
│     ▼              ▼                    ▼                           │
│  ┌──────────┐  ┌──────────┐       ┌──────────┐                     │
│  │  ARIMA   │  │ Prophet  │       │   LSTM   │                     │
│  │          │  │          │       │          │                     │
│  │Classical │  │Facebook  │       │ Deep     │                     │
│  │stats     │  │decompose │       │ Learning │                     │
│  │ADF test  │  │Bayesian  │       │Sequence  │                     │
│  │ACF/PACF  │  │intervals │       │learning  │                     │
│  └────┬─────┘  └────┬─────┘       └────┬─────┘                    │
│       └─────────────┴────────────────── ┘                           │
│                          │                                           │
│               60-Day Forecast Comparison                             │
│               RMSE · MAE · Visual overlay                            │
└──────────────────────────────────────────────────────────────────────┘
```

---

## ◈ Model Comparison

| Model | RMSE (kWh) | MAE (kWh) | Strength |
|---|---|---|---|
| ARIMA | ~15–25 | ~12–20 | Interpretable, principled |
| **Prophet** | **~10–18** | **~8–15** | Auto seasonality, robust |
| LSTM | ~12–22 | ~9–18 | Non-linear, long-range |

---

## ◈ The Data — Built From Physics

```python
consumption = trend + weekly_seasonality + yearly_seasonality + noise
# 3 years daily (2020-01-01 → 2022-12-31)
# Trend:   +0.05 kWh/day linear growth
# Weekly:  lower on weekends
# Yearly:  peak winter, trough summer (cosine wave)
# Noise:   Gaussian day-to-day variation
```

No external download required — the synthetic generator runs automatically.

---

## ◈ ARIMA Deep Dive

- Augmented Dickey-Fuller test → confirms non-stationarity
- First-order differencing → achieves stationarity
- ACF + PACF plots → identify lag structure
- ARIMA(2,1,2) fitted via statsmodels
- 60-day out-of-sample forecast

**Why ARIMA?** Linear combinations of past values and forecast errors. Highly interpretable. Fails on non-linear dynamics.

---

## ◈ Prophet Deep Dive

- Data reformatted to `ds`/`y` convention
- Yearly + weekly seasonality enabled
- Uncertainty intervals built-in
- Component plots: trend / weekly / yearly decomposition

**Why Prophet?** Developed at Meta. Minimal tuning. Handles missing data. Calibrated uncertainty. Best for stakeholder communication.

---

## ◈ LSTM Deep Dive

```
Input → LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1)
```

- 30-day look-back window
- Min-max scaling → inverse transform at inference
- 30 epochs, batch size 32

**Why LSTM?** Captures non-linear temporal dependencies. Adapts to structural breaks. Highest data requirements, lowest interpretability.

---

## ◈ Notebook Walkthrough

| Cell | Contents |
|---|---|
| 1 | Introduction & problem statement |
| 2 | Imports |
| 3 | Synthetic data generation |
| 4 | EDA: rolling averages, seasonal decomposition |
| 5 | Stationarity testing + differencing |
| 6 | ARIMA model + forecast |
| 7 | Prophet model + component plots |
| 8 | LSTM model + training |
| 9 | Three-model comparison chart + metrics table |
| 10 | Conclusions & next steps |

---

## ◈ Quick Start

```bash
# 1. Clone
git clone https://github.com/isamkhan1809/time-series-forecasting.git
cd time-series-forecasting

# 2. Install
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Launch
jupyter notebook time_series_forecasting.ipynb
# Run All Cells — no data download needed
```

> **Apple Silicon:** replace `tensorflow==2.15.0` with `tensorflow-macos` + `tensorflow-metal`

---

## ◈ Project Structure

```
time-series-forecasting/
├── time_series_forecasting.ipynb  ← Full pipeline (10 cells)
├── requirements.txt
├── data/                          ← External dataset placeholder
└── README.md
```

---

<div align="center">

**Three models enter. One future emerges.**

*MIT License*

</div>
