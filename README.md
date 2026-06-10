<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Time%20Series%20Forecasting&fontSize=58&fontColor=fff&animation=twinkling&fontAlignY=35&desc=ARIMA%20vs%20Prophet%20vs%20LSTM%20%E2%80%94%20Three%20Models%2C%20One%20Future&descAlignY=60&descSize=18" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-9B59B6?style=for-the-badge&logo=python&logoColor=white&labelColor=0D0D0D)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white&labelColor=0D0D0D)](https://tensorflow.org)
[![Prophet](https://img.shields.io/badge/Prophet-Meta-9B59B6?style=for-the-badge&logoColor=white&labelColor=0D0D0D)](https://facebook.github.io/prophet)
[![License](https://img.shields.io/badge/License-MIT-9B59B6?style=for-the-badge&labelColor=0D0D0D)](LICENSE)

<br/>

<a href="https://github.com/isamkhan1809/time-series-forecasting">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=22&pause=1000&color=9B59B6&center=true&vCenter=true&width=700&lines=ARIMA+vs+Prophet+vs+LSTM+%E2%80%94+Head+to+Head;60-Day+Energy+Consumption+Forecast;Classical+Stats+vs+Deep+Learning;3+Years+of+Data.+One+Winner." alt="Typing SVG" />
</a>

</div>

---

<br/>

<div align="center">

```
  ╔══════════════════════════════════════════════════════════════╗
  ║                                                              ║
  ║   ARIMA sees patterns in the past.                          ║
  ║   Prophet decomposes the seasons.                           ║
  ║   LSTM learns the non-linear truth.                         ║
  ║                                                              ║
  ║       Three paradigms. One 60-day forecast. Who wins?       ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
```

</div>

<br/>

## `>_ The Story`

> *Energy consumption is one of the most structured time series in existence — a long-term trend, a weekly rhythm, a yearly seasonal swing. It's the perfect arena to pit three forecasting paradigms against each other.*
>
> *ARIMA — the classical statistician — tests stationarity, differences the series, and fits a linear model to its own lags.*
>
> *Prophet — Meta's decomposition engine — separates trend, weekly, and yearly seasonality automatically.*
>
> *LSTM — the deep learner — reads 30 days of history and predicts the next, learning non-linear dependencies the others can't see.*
>
> *All three are trained. All three are evaluated. This notebook shows you exactly where each one wins and where it breaks.*

<br/>

## `>_ The Data`

```python
# 3 years of synthetic daily energy consumption
consumption = trend + weekly_seasonality + yearly_seasonality + noise

# 2020-01-01 → 2022-12-31 (1,096 data points)
# Trend:   +0.05 kWh/day  — slow linear growth
# Weekly:  lower on weekends
# Yearly:  peak in winter, trough in summer (cosine wave)
# Noise:   Gaussian day-to-day variation
```

No external download required — the generator runs automatically in Cell 3.

<br/>

## `>_ Architecture`

```
┌─────────────────────────────────────────────────────────────┐
│                  FORECASTING PIPELINE                       │
│                                                             │
│  3 Years Daily Energy Data                                  │
│  ─────────────────────────                                  │
│  Trend + Seasonality + Noise                                │
│                 │                                           │
│     ┌───────────┼───────────┐                              │
│     ▼           ▼           ▼                              │
│  ┌────────┐  ┌─────────┐  ┌──────┐                         │
│  │ ARIMA  │  │ Prophet │  │ LSTM │                         │
│  │        │  │         │  │      │                         │
│  │ ADF    │  │ ds/y    │  │ 30d  │                         │
│  │ ACF    │  │ format  │  │ look │                         │
│  │ PACF   │  │ yearly+ │  │ back │                         │
│  │ (2,1,2)│  │ weekly  │  │ LSTM │                         │
│  └───┬────┘  └────┬────┘  │ (64) │                         │
│      │            │       │ LSTM │                         │
│      │            │       │ (32) │                         │
│      │            │       └──┬───┘                         │
│      └────────────┴──────────┘                             │
│                   │                                        │
│           60-Day Forecast Comparison                        │
│           RMSE · MAE · Visual Overlay                       │
└─────────────────────────────────────────────────────────────┘
```

<br/>

## `>_ Results`

<div align="center">

| Model | RMSE (kWh) | MAE (kWh) | Best At |
|---|---|---|---|
| ARIMA | ~15–25 | ~12–20 | Interpretable, linear dynamics |
| **Prophet** | **~10–18** | **~8–15** | Auto seasonality, stakeholder reports |
| LSTM | ~12–22 | ~9–18 | Non-linear, structural breaks |

</div>

<br/>

## `>_ LSTM Architecture`

```
Input (30-day sequence)
  → LSTM(64) → Dropout(0.2)
  → LSTM(32) → Dropout(0.2)
  → Dense(1)

30 epochs · batch size 32 · min-max scaled
Inverse transform applied at inference
```

<br/>

## `>_ Notebook Walkthrough`

| Cell | Contents |
|---|---|
| 1–3 | Introduction, imports, synthetic data generation |
| 4–5 | EDA, seasonal decomposition, stationarity testing |
| 6 | ARIMA — ACF/PACF, fitting, 60-day forecast |
| 7 | Prophet — fitting, component plots, forecast |
| 8 | LSTM — training, inference, inverse transform |
| 9–10 | Three-model comparison chart + conclusions |

<br/>

## `>_ Get Running`

```bash
# Clone
git clone https://github.com/isamkhan1809/time-series-forecasting.git
cd time-series-forecasting

# Install
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Launch
jupyter notebook time_series_forecasting.ipynb
# Run All Cells — no download needed
```

> **Apple Silicon:** replace `tensorflow==2.15.0` with `tensorflow-macos` + `tensorflow-metal`

<br/>

## `>_ Tech Stack`

<div align="center">

| Layer | Technology |
|---|---|
| **Classical** | statsmodels (ARIMA) |
| **Decomposition** | Prophet (Meta) |
| **Deep Learning** | TensorFlow / Keras (LSTM) |
| **Data** | pandas, numpy |
| **Visualisation** | matplotlib, seaborn |

</div>

<br/>

## `>_ Project Structure`

```
time-series-forecasting/
├── time_series_forecasting.ipynb  ← Full 10-cell pipeline
├── requirements.txt
└── data/                          ← Optional external datasets
```

<br/>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer&animation=twinkling" width="100%"/>

<br/>

*Three models enter. One future emerges.*
*ARIMA · Prophet · LSTM — compared head to head.*

<br/>

[![GitHub](https://img.shields.io/badge/github-isamkhan1809-9B59B6?style=for-the-badge&logo=github&logoColor=white&labelColor=0D0D0D)](https://github.com/isamkhan1809)

</div>
