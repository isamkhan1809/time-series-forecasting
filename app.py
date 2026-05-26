import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── Global state ────────────────────────────────────────────────────────────
historical_data: pd.Series | None = None
arima_model = None
ets_model = None
forecast_cache: dict = {}
summary_cache: dict = {}


# ── Data generation ──────────────────────────────────────────────────────────
def generate_energy_data() -> pd.Series:
    """Generate 2 years of synthetic daily energy consumption (kWh)."""
    np.random.seed(42)
    n_days = 730  # ~2 years
    start = datetime(2022, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]

    t = np.arange(n_days)

    # Long-term upward trend
    trend = 200 + 0.08 * t

    # Weekly seasonality: higher on weekdays, lower on weekends
    weekly = 20 * np.sin(2 * np.pi * t / 7 + 1.5)

    # Yearly seasonality: peak in winter / summer, dip in spring / autumn
    yearly = 40 * np.cos(2 * np.pi * t / 365.25)

    # Random noise
    noise = np.random.normal(0, 15, n_days)

    values = trend + weekly + yearly + noise
    values = np.clip(values, 50, None)  # no negatives

    return pd.Series(values, index=pd.DatetimeIndex(dates), name="consumption")


# ── Model fitting ─────────────────────────────────────────────────────────────
def fit_models(series: pd.Series):
    """Fit ARIMA(2,1,2) and Holt-Winters ETS on the full series."""
    # ARIMA
    arima = ARIMA(series, order=(2, 1, 2))
    arima_fit = arima.fit()

    # Exponential Smoothing – Holt-Winters with additive weekly seasonality
    ets = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add",
        seasonal_periods=7,
        initialization_method="estimated",
    )
    ets_fit = ets.fit(optimized=True)

    return arima_fit, ets_fit


# ── Metrics helpers ───────────────────────────────────────────────────────────
def compute_metrics(series: pd.Series, fitted_values: pd.Series) -> dict:
    common_idx = series.index.intersection(fitted_values.index)
    y_true = series.loc[common_idx].values
    y_pred = fitted_values.loc[common_idx].values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
    mae = float(mean_absolute_error(y_true[mask], y_pred[mask]))
    return {"rmse": round(rmse, 3), "mae": round(mae, 3)}


# ── Forecast builders ─────────────────────────────────────────────────────────
def build_arima_forecast(horizon: int) -> dict:
    fc = arima_model.get_forecast(steps=horizon)
    mean_fc = fc.predicted_mean
    ci = fc.conf_int(alpha=0.10)  # 90 % CI

    metrics = compute_metrics(historical_data, arima_model.fittedvalues)
    metrics["model_name"] = "ARIMA(2,1,2)"

    forecast_list = [
        {
            "date": str(mean_fc.index[i].date()),
            "value": round(float(mean_fc.iloc[i]), 2),
            "lower": round(float(ci.iloc[i, 0]), 2),
            "upper": round(float(ci.iloc[i, 1]), 2),
        }
        for i in range(horizon)
    ]
    return {"forecast": forecast_list, "metrics": metrics}


def build_ets_forecast(horizon: int) -> dict:
    fc_values = ets_model.forecast(horizon)
    last_date = historical_data.index[-1]
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=horizon, freq="D")

    metrics = compute_metrics(historical_data, ets_model.fittedvalues)
    metrics["model_name"] = "Holt-Winters ETS"

    forecast_list = [
        {
            "date": str(future_dates[i].date()),
            "value": round(float(fc_values.iloc[i]), 2),
            "lower": round(float(fc_values.iloc[i] * 0.90), 2),
            "upper": round(float(fc_values.iloc[i] * 1.10), 2),
        }
        for i in range(horizon)
    ]
    return {"forecast": forecast_list, "metrics": metrics}


# ── Startup initialisation ────────────────────────────────────────────────────
def startup():
    global historical_data, arima_model, ets_model, forecast_cache, summary_cache

    print("Generating synthetic energy data …")
    historical_data = generate_energy_data()

    print("Fitting ARIMA(2,1,2) …")
    print("Fitting Holt-Winters ExponentialSmoothing …")
    arima_model, ets_model = fit_models(historical_data)

    print("Pre-computing forecast cache …")
    for horizon in (30, 60, 90):
        forecast_cache[f"arima_{horizon}"] = build_arima_forecast(horizon)
        forecast_cache[f"ets_{horizon}"] = build_ets_forecast(horizon)

    # Summary statistics
    avg_val = float(historical_data.mean())
    peak_day = str(historical_data.idxmax().date())
    min_day = str(historical_data.idxmin().date())

    # Simple trend: compare last 30-day mean vs first 30-day mean
    first_mean = float(historical_data.iloc[:30].mean())
    last_mean = float(historical_data.iloc[-30:].mean())
    trend = "Upward" if last_mean > first_mean * 1.01 else ("Downward" if last_mean < first_mean * 0.99 else "Stable")

    summary_cache = {
        "avg_consumption": round(avg_val, 1),
        "peak_day": peak_day,
        "min_day": min_day,
        "total_days": len(historical_data),
        "trend_direction": trend,
    }

    print("Startup complete. App is ready.")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/forecast", methods=["POST"])
def forecast():
    body = request.get_json(force=True) or {}
    model_key = body.get("model", "arima").lower()
    horizon = int(body.get("horizon", 30))

    if model_key not in ("arima", "ets"):
        return jsonify({"error": "model must be 'arima' or 'ets'"}), 400
    if horizon not in (30, 60, 90):
        return jsonify({"error": "horizon must be 30, 60, or 90"}), 400

    cache_key = f"{model_key}_{horizon}"
    cached = forecast_cache.get(cache_key)
    if cached is None:
        return jsonify({"error": "Forecast not available"}), 500

    # Last 90 days of history for the chart
    hist_slice = historical_data.iloc[-90:]
    historical_list = [
        {"date": str(dt.date()), "value": round(float(val), 2)}
        for dt, val in zip(hist_slice.index, hist_slice.values)
    ]

    return jsonify(
        {
            "historical": historical_list,
            "forecast": cached["forecast"],
            "metrics": cached["metrics"],
        }
    )


@app.route("/api/summary")
def summary():
    return jsonify(summary_cache)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    startup()
    app.run(debug=False, port=5000)
