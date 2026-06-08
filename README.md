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

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&pause=1000&color=9B59B6&center=true&vCenter=true&width=700&lines=ARIMA.+Prophet.+LSTM.+Three+Models.+One+Future.+%F0%9F%94%AE;60-Day+Energy+Consumption+Forecast;Classical+Stats+vs+Deep+Learning+%E2%80%94+Head+to+Head;3+Years+of+Data+%7C+Full+Evaluation+Pipeline" alt="Typing SVG" />

<img src="https://media.giphy.com/media/du3J3cXyzhj75IOgvA/giphy.gif" width="360" />

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Prophet](https://img.shields.io/badge/Prophet-Meta-4267B2?style=for-the-badge)](https://facebook.github.io/prophet)
[![statsmodels](https://img.shields.io/badge/ARIMA-statsmodels-blue?style=for-the-badge)](https://www.statsmodels.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> **A complete time series forecasting project that trains ARIMA, Prophet, and LSTM on 3 years of energy consumption data — then pits them head-to-head on a 60-day forecast.**

</div>

---

## ◈ Three Paradigms. One Problem.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    FORECASTING ARCHITECTURE                          │
│                                                                      │
│   3 Years Daily Energy Data (kWh)                                    │
│   Trend + Weekly + Yearly seasonality + Gaussian Noise               │
│              │                                                       │
│     ┌────────┴──────────────────────────┐                           │
│     ▼              ▼                    ▼                           │
│  ┌──────────┐  ┌──────────┐       ┌──────────┐                     │
│  │  ARIMA   │  │ Prophet  │       │   LSTM   │                     │
│  │Classical │  │Facebook  │       │ Deep     │                     │
│  │ADF test  │  │decompose │       │Learning  │                     │
│  │ACF/PACF  │  │Bayesian  │       │Sequence  │                     │
│  └────┬─────┘  └────┬─────┘       └────┬─────┘                    │
│       └─────────────┴────────────────── ┘                           │
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

## ◈ The Data — Synthetic But Realistic

```python
consumption = trend + weekly_seasonality + yearly_seasonality + noise
# 2020-01-01 → 2022-12-31 (3 years daily)
# Trend:   +0.05 kWh/day linear growth
# Weekly:  lower consumption on weekends
# Yearly:  peak winter, trough summer (cosine wave)
# Noise:   Gaussian day-to-day variation
```

No external download required — the generator runs automatically.

---

## ◈ Model Deep Dives

**ARIMA** — ADF stationarity test → first-order differencing → ACF/PACF → ARIMA(2,1,2). Highly interpretable. Fails on non-linear dynamics.

**Prophet** — Meta's decomposition model. `ds`/`y` format. Yearly + weekly seasonality. Calibrated uncertainty intervals. Best for stakeholder communication.

**LSTM**
```
Input → LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1)
30-day look-back · 30 epochs · batch size 32 · min-max scaled
```

---

## ◈ Notebook Walkthrough

| Cell | Contents |
|---|---|
| 1–3 | Intro, imports, synthetic data |
| 4–5 | EDA, seasonal decomposition, stationarity |
| 6 | ARIMA model + forecast |
| 7 | Prophet model + component plots |
| 8 | LSTM model + training |
| 9–10 | Three-model comparison + conclusions |

---

## ◈ Quick Start

```bash
git clone https://github.com/isamkhan1809/time-series-forecasting.git
cd time-series-forecasting
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter notebook time_series_forecasting.ipynb
# Run All Cells — no download needed
```

> **Apple Silicon:** use `tensorflow-macos` + `tensorflow-metal` instead of `tensorflow==2.15.0`

---

## ◈ Project Structure

```
time-series-forecasting/
├── time_series_forecasting.ipynb
├── requirements.txt
├── data/
└── README.md
```

---

<div align="center">

**Three models enter. One future emerges.**

*MIT License*

<br/>

Interested in forecasting, time series modelling, or energy analytics?<br/>
Let's connect — built by <a href="https://github.com/isamkhan1809">Isam Khan</a> &nbsp;|&nbsp;
<a href="https://linkedin.com/in/isam-khan-3a1260292"><img src="https://img.shields.io/badge/-LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white&labelColor=000000"/></a>
<a href="https://isamkhan.com"><img src="https://img.shields.io/badge/-isamkhan.com-00D9FF?style=flat-square&logo=googlechrome&logoColor=white&labelColor=000000"/></a>

</div>
