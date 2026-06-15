
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import numpy as np
import pandas as pd


def global_mean(train_df, val_df, target_col):
    """
    Forecast = Durchschnitt der jeweiligen series_id im Training.
    """
    val_pred = val_df.copy()

    series_mean = train_df.groupby("series_id")[target_col].mean()

    val_pred["prediction"] = val_pred["series_id"].map(series_mean)

    fallback = train_df[target_col].mean()
    val_pred["prediction"] = val_pred["prediction"].fillna(fallback)

    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred


def seasonal_naive(train_df, val_df, target_col):
    """
    Forecast = Wert von vor 7 Tagen.
    """
    val_pred = val_df.copy()

    val_start = val_pred["day_idx"].min()

    last_week = train_df[
        train_df["day_idx"].between(val_start - 7, val_start - 1)
    ][["series_id", "day_idx", target_col]].copy()

    last_week["forecast_day"] = last_week["day_idx"] + 7

    last_week = last_week.rename(columns={target_col: "prediction"})

    val_pred = val_pred.merge(
        last_week[["series_id", "forecast_day", "prediction"]],
        left_on=["series_id", "day_idx"],
        right_on=["series_id", "forecast_day"],
        how="left"
    )

    val_pred = val_pred.drop(columns=["forecast_day"], errors="ignore")

    fallback = train_df.groupby("series_id")[target_col].mean()

    val_pred["prediction"] = val_pred["prediction"].fillna(
        val_pred["series_id"].map(fallback)
    )

    global_fallback = train_df[target_col].mean()
    val_pred["prediction"] = val_pred["prediction"].fillna(global_fallback)

    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred


def rolling_28d(train_df, val_df, target_col):
    """
    Forecast = Durchschnitt der letzten 28 Trainingstage pro series_id.
    """
    val_pred = val_df.copy()

    roll28 = train_df.groupby("series_id")[target_col].apply(
        lambda x: x.tail(28).mean()
    )

    val_pred["prediction"] = val_pred["series_id"].map(roll28)

    fallback = train_df.groupby("series_id")[target_col].mean()

    val_pred["prediction"] = val_pred["prediction"].fillna(
        val_pred["series_id"].map(fallback)
    )

    global_fallback = train_df[target_col].mean()
    val_pred["prediction"] = val_pred["prediction"].fillna(global_fallback)

    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred


def exponential_smoothing(train_df, val_df, target_col, seasonal_periods=7):
    """
    Forecast via Holt-Winters Exponential Smoothing pro series_id.

    Für jede series_id wird auf den (nach day_idx sortierten)
    Trainingsdaten ein ExponentialSmoothing-Modell gefittet und
    für den benötigten Horizont extrapoliert.

    Fallback-Kette (je nach Datenmenge / Fit-Erfolg):
    1. additiver Trend (gedämpft) + additive Saisonalität (seasonal_periods)
    2. additiver Trend (gedämpft), keine Saisonalität
    3. einfaches Level-Smoothing (SES), kein Trend, keine Saison
    4. series_id Durchschnitt
    5. globaler Durchschnitt
    """
    val_pred = val_df.copy()

    global_fallback = train_df[target_col].mean()
    series_fallback = train_df.groupby("series_id")[target_col].mean()

    predictions = {}

    for series_id, val_group in val_pred.groupby("series_id"):
        train_group = train_df[train_df["series_id"] == series_id].sort_values("day_idx")

        val_days = val_group["day_idx"].sort_values().values

        if train_group.empty:
            fb = global_fallback
            for day_idx in val_days:
                predictions[(series_id, day_idx)] = fb
            continue

        y = train_group[target_col].astype(float).values
        n = len(y)

        train_max_day = train_group["day_idx"].max()
        max_horizon = int(val_days.max() - train_max_day)
        max_horizon = max(max_horizon, 1)

        forecast = None

        # Versuch 1: Trend (gedämpft) + Saisonalität
        if n >= 2 * seasonal_periods:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ExponentialSmoothing(
                        y,
                        trend="add",
                        damped_trend=True,
                        seasonal="add",
                        seasonal_periods=seasonal_periods,
                        initialization_method="estimated",
                    ).fit()
                forecast = model.forecast(max_horizon)
            except Exception:
                forecast = None

        # Versuch 2: nur Trend (gedämpft)
        if forecast is None and n >= 2:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ExponentialSmoothing(
                        y,
                        trend="add",
                        damped_trend=True,
                        seasonal=None,
                        initialization_method="estimated",
                    ).fit()
                forecast = model.forecast(max_horizon)
            except Exception:
                forecast = None

        # Versuch 3: simple exponential smoothing (kein Trend, keine Saison)
        if forecast is None and n >= 1:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ExponentialSmoothing(
                        y,
                        trend=None,
                        seasonal=None,
                        initialization_method="estimated",
                    ).fit()
                forecast = model.forecast(max_horizon)
            except Exception:
                forecast = None

        # Fallback: series_id / globaler Durchschnitt
        if forecast is None:
            fb = series_fallback.get(series_id, global_fallback)
            forecast = np.full(max_horizon, fb)

        for day_idx in val_days:
            step = int(day_idx - train_max_day) - 1  # 0-indexiert
            if 0 <= step < len(forecast):
                pred_value = forecast[step]
            else:
                pred_value = series_fallback.get(series_id, global_fallback)
            predictions[(series_id, day_idx)] = pred_value

    val_pred["prediction"] = val_pred.apply(
        lambda row: predictions.get((row["series_id"], row["day_idx"]), global_fallback),
        axis=1,
    )

    val_pred["prediction"] = val_pred["prediction"].fillna(global_fallback)
    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred