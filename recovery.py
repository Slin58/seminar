# recovery.py contains all recovery methods we implemented
# recovery methods need to add column: "recovered_daily_sales" to history

# TODO bei recovery: recovern von gleichen city_id, store_id, management_group_id, first_category_id, second_category_id, third_category_id, product_id ??

import numpy as np
import pandas as pd
import os
import time

from scipy.optimize import minimize
from scipy.special import ndtr, log_ndtr
from scipy.stats import norm

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.seasonal import STL

import lightgbm as lgb
from xgboost import XGBRegressor


# Einfache Imputation Methoden:
def random_sampling(history, op_sales_masked, outside_slice, rng): # Simple recovery: random pool sampling
    imputed = op_sales_masked.copy()
    imputed_count = 0
    for h in range(16):
        col = imputed[:, h] # alle Werte der Stunde h
        mask = np.isnan(col) # Maske für fehlende Werte in Stunde h (True für fehlende Werte, False für vorhandene Werte)
        n_miss = mask.sum() 
        if n_miss > 0:
            pool = col[~mask] # alle Werte der Stunde h, die nicht fehlen
            imputed[mask, h] = np.maximum(0, rng.choice(pool, size=n_miss, replace=True))

            imputed_count += n_miss

    # Rebuild corrected daily target
    recovered_sum = np.nansum(imputed, axis=1)
    recovered_daily = recovered_sum + outside_slice

    history["recovered_daily_sales_random_sampling"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_random_sampling'].mean():.4f}")

def global_mean(history, op_sales_masked, outside_slice):  # globaler Durchschnitt
    imputed = op_sales_masked.copy()
    # Durchschnitt über alle sichtbaren Stundenwerte
    mean_value = np.nanmean(imputed)
    # Nur NaN-Werte ersetzen
    mask = np.isnan(imputed)
    imputed_count = mask.sum()
    imputed[mask] = mean_value
    # Rebuild corrected daily target
    recovered_sum = np.nansum(imputed, axis=1)
    recovered_daily = outside_slice + recovered_sum
    history["recovered_daily_sales_global_mean"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Global mean used: {mean_value:.4f}")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_global_mean'].mean():.4f}")

def per_series_mean(history): # Durchschnitt derselben series_id - Nils
    recovered_daily = history["sale_amount"].where(history["is_censored"] == 0, np.nan)

    series_mean = recovered_daily.groupby(history["series_id"]).transform("mean")

    recovered_daily = recovered_daily.fillna(series_mean)

    history["recovered_daily_sales_per_series_mean"] = recovered_daily

    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_per_series_mean'].mean():.4f}")

def hour_per_series_mean(history, op_sales_masked, outside_slice): # Durchschnitt derselben series_id & derselben Stunde - Nils
    imputed = op_sales_masked.copy()

    # ---------- SERIES IDS ----------
    series_codes, unique_series = pd.factorize(history["series_id"], sort=False)
    n_series = len(unique_series)
    n_rows, n_hours = imputed.shape

    # ---------- GLOBAL HOURLY MEANS (fallback) ----------
    global_hourly_means = np.nanmean(imputed, axis=0)  # (n_hours,)

    # ---------- PER-SERIES HOURLY MEANS (vectorized) ----------
    valid = ~np.isnan(imputed)  # (n_rows, n_hours)

    sums   = np.zeros((n_series, n_hours))
    counts = np.zeros((n_series, n_hours))

    np.add.at(sums,   series_codes, np.where(valid, imputed, 0))
    np.add.at(counts, series_codes, valid.astype(float))

    with np.errstate(invalid="ignore"):
        series_hourly_means = np.where(counts > 0, sums / counts, global_hourly_means)

    # ---------- IMPUTE ----------
    nan_mask = np.isnan(imputed)
    imputed[nan_mask] = series_hourly_means[series_codes][nan_mask]
    imputed = np.maximum(imputed, 0)
    imputed_count = nan_mask.sum()

    # ---------- REBUILD DAILY SALES ----------
    recovered_daily = np.nansum(imputed, axis=1) + outside_slice
    history["recovered_daily_sales_hour_per_series_mean"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_hour_per_series_mean'].mean():.4f}")

def weekday_per_series_mean():
    return

def weekday_mean(history, op_sales_masked, outside_slice): # Durchschnitt gleicher Wochentage - Laura
    history["weekday"] = history["dt"].dt.weekday

    imputed = op_sales_masked.copy()
    weekdays = history["weekday"].values
    n_hours = imputed.shape[1]

    # global fallback pro Stunde
    global_hour_means = np.nanmean(imputed, axis=0)

    imputed_count = np.isnan(imputed).sum()

    for h in range(n_hours):
        col = imputed[:, h]

        for wd in range(7):
            wd_mask = weekdays == wd

            mean_value = np.nanmean(col[wd_mask])

            if np.isnan(mean_value):
                mean_value = global_hour_means[h]

            nan_mask = wd_mask & np.isnan(col)

            col[nan_mask] = mean_value

        imputed[:, h] = col

    recovered_sum = np.nansum(imputed, axis=1)
    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_weekday_mean"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_weekday_mean'].mean():.4f}")

def weekday_daily_mean(history):  # Durchschnitt desselben Wochentags - Nils
    recovered_daily = history["sale_amount"].where(history["is_censored"] == 0, np.nan)

    dayofweek = pd.to_datetime(history["dt"]).dt.dayofweek
    hour = pd.to_datetime(history["dt"]).dt.hour

    weekday_daily_mean_key = dayofweek.astype(str) + "_" + hour.astype(str)
    weekday_daily_mean = recovered_daily.groupby(weekday_daily_mean_key).transform("mean")

    recovered_daily = recovered_daily.fillna(weekday_daily_mean)

    history["recovered_daily_sales_weekday_daily_mean"] = recovered_daily

    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_weekday_daily_mean'].mean():.4f}")


def hourly_mean(history, op_sales_masked, outside_slice): # Durchschnitt der gleichen Stunde - Nils
    imputed = op_sales_masked.copy()

    global_hourly_means = np.nanmean(imputed, axis=0)

    nan_mask = np.isnan(imputed)

    replacement_values = np.tile(global_hourly_means, (imputed.shape[0], 1))

    imputed[nan_mask] = replacement_values[nan_mask]

    imputed = np.maximum(imputed, 0)

    imputed_count = nan_mask.sum()

    recovered_sum = np.nansum(imputed, axis=1)

    recovered_daily = recovered_sum + outside_slice

    history["recovered_daily_sales_hourly_mean"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_hourly_mean'].mean():.4f}")


# Moving averages: - Laura
def rolling_mean(history, op_sales_masked, outside_slice, window=7): # SMA / Rolling Mean - Laura
    imputed = op_sales_masked.copy()
    imputed_count = np.isnan(imputed).sum()

    for h in range(imputed.shape[1]):
        s = pd.Series(imputed[:, h])

        roll = s.rolling(window=window, min_periods=1).mean()

        mask = s.isna()

        # falls am Anfang noch nichts vorhanden ist
        fallback = s.mean()

        imputed[mask.values, h] = roll[mask].fillna(fallback).values

    recovered_sum = np.nansum(imputed, axis=1)
    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_rolling_mean"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Window size used: {window}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_rolling_mean'].mean():.4f}")

def exponential_moving_average(history, op_sales_masked, outside_slice, alpha=0.3): # EMA - Laura
    imputed = op_sales_masked.copy()
    imputed_count = np.isnan(imputed).sum()

    for h in range(imputed.shape[1]):
        s = pd.Series(imputed[:, h])

        ema = s.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()

        mask = s.isna()

        fallback = s.mean()

        imputed[mask.values, h] = ema[mask].fillna(fallback).values

    recovered_sum = np.nansum(imputed, axis=1)
    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_exponential_moving_average"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Alpha used: {alpha}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_exponential_moving_average'].mean():.4f}")

def exponential_moving_average_series(history, op_sales_masked, outside_slice, alpha=0.3): # lädt ca 2,5 min 
    imputed = op_sales_masked.copy()
    imputed_count = np.isnan(imputed).sum()

    series_ids = history["series_id"].values
    n_hours = imputed.shape[1]

    for h in range(n_hours):
        s = pd.Series(imputed[:, h])

        ema = (
            s.groupby(series_ids)
            .transform(lambda x: x.ewm(alpha=alpha, adjust=False, ignore_na=True).mean())
        )

        mask = s.isna()
        fallback = s.mean()

        imputed[mask.values, h] = ema[mask].fillna(fallback).values

    imputed = np.maximum(imputed, 0)

    recovered_sum = np.nansum(imputed, axis=1)
    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_exponential_moving_average_series"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Alpha used: {alpha}")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_exponential_moving_average_series'].mean():.4f}")

# Zeitreihenbasierte Recovery-Methoden: - Laura


def interpolation_linear(history, op_sales_masked, outside_slice):  # Lineare Interpolation- Laura
    # Interpolieren zwischen zwei Werten (letzter bekannter Wert und nächster bekannter Wert)

    imputed = op_sales_masked.copy()

    imputed_count = np.isnan(imputed).sum()

    n_hours = imputed.shape[1]

    # Jede Stunde einzeln
    for h in range(n_hours):

        s = pd.Series(imputed[:, h])

        # Linear interpolieren
        interpolated = s.interpolate(
            method="linear",
            limit_direction="both"
        )

        # Fallback falls komplett NaN
        fallback = s.mean()

        interpolated = interpolated.fillna(fallback)

        imputed[:, h] = interpolated.values

    # Sicherheit
    imputed = np.maximum(imputed, 0)

    # Rebuild corrected daily target
    recovered_sum = np.nansum(imputed, axis=1)

    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_interpolation_linear"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(
        f"Mean recovered sales: "
        f"{history['recovered_daily_sales_interpolation_linear'].mean():.4f}"
    )


def interpolation_spline(history, op_sales_masked, outside_slice, order=3):  # Spline-Interpolation  Laura

    imputed = op_sales_masked.copy()

    imputed_count = np.isnan(imputed).sum()
    n_hours = imputed.shape[1]

    for h in range(n_hours):

        s = pd.Series(imputed[:, h])

        # Spline braucht genug bekannte Werte
        if s.notna().sum() > order:
            interpolated = s.interpolate(
                method="spline",
                order=order,
                limit_direction="both"
            )
        else:
            interpolated = s.copy()

        # Fallback für übrige NaNs
        fallback = s.mean()
        interpolated = interpolated.fillna(fallback)

        imputed[:, h] = interpolated.values

    imputed = np.maximum(imputed, 0)

    recovered_sum = np.nansum(imputed, axis=1)
    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_interpolation_spline"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Spline order used: {order}")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_interpolation_spline'].mean():.4f}")

def interpolation_spline_series(history, op_sales_masked, outside_slice, order=3): # TODO -> Nils durchlaufen lassen
    imputed = op_sales_masked.copy()
    imputed_count = np.isnan(imputed).sum()

    series_ids = history["series_id"].values
    n_hours = imputed.shape[1]

    for h in range(n_hours):
        s = pd.Series(imputed[:, h])

        # Interpolation nur innerhalb derselben series_id
        interpolated = (
            s.groupby(series_ids)
             .transform(
                 lambda x: x.interpolate(
                     method="spline",
                     order=order,
                     limit_direction="both"
                 ) if x.notna().sum() > order else x
             )
        )

        # Fallback pro Stunde
        fallback = s.mean()
        interpolated = interpolated.fillna(fallback)

        imputed[:, h] = interpolated.values

    imputed = np.maximum(imputed, 0)

    recovered_sum = np.nansum(imputed, axis=1)
    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_interpolation_spline_series"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Spline order used: {order}")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_interpolation_spline_series'].mean():.4f}")

def interpolation_polynomial(history, op_sales_masked, outside_slice, order=2):  # Polynomial-Interpolation - Laura

    imputed = op_sales_masked.copy()

    imputed_count = np.isnan(imputed).sum()

    n_hours = imputed.shape[1]

    # Jede Stunde einzeln
    for h in range(n_hours):

        s = pd.Series(imputed[:, h])

        # Polynomial braucht genug bekannte Werte
        if s.notna().sum() > order:

            interpolated = s.interpolate(method="polynomial", order=order, limit_direction="both")

        else:
            interpolated = s.copy()

        # Fallback falls noch NaNs existieren
        fallback = s.mean()

        interpolated = interpolated.fillna(fallback)

        imputed[:, h] = interpolated.values

    # Sicherheit
    imputed = np.maximum(imputed, 0)

    # Rebuild corrected daily target
    recovered_sum = np.nansum(imputed, axis=1)

    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_interpolation_polynomial"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Polynomial order used: {order}")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(
        f"Mean recovered sales: "
        f"{history['recovered_daily_sales_interpolation_polynomial'].mean():.4f}"
    )

def kalman_smoothing(history, op_sales_masked, outside_slice):  # Kalman Smoothing / State Space - Laura 

    imputed = op_sales_masked.copy()

    imputed_count = np.isnan(imputed).sum()
    n_hours = imputed.shape[1]

    for h in range(n_hours):

        s = pd.Series(imputed[:, h]).astype(float)

        # Falls zu wenig echte Werte vorhanden sind: Fallback auf Stundenmittel
        if s.notna().sum() < 10:
            filled = s.fillna(s.mean())
        else:
            try:
                # Local level model = einfaches State-Space-Modell
                model = UnobservedComponents(s, level="local level")

                result = model.fit(disp=False)

                # Smoothed states als geschätzte Werte
                smoothed = result.smoothed_state[0]

                filled = s.copy()
                filled[s.isna()] = smoothed[s.isna()]

                # Falls noch NaNs übrig bleiben
                filled = filled.fillna(s.mean())

            except Exception:
                # Falls das Modell nicht konvergiert
                filled = s.fillna(s.mean())

        imputed[:, h] = filled.values

    imputed = np.maximum(imputed, 0)

    recovered_sum = np.nansum(imputed, axis=1)
    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_kalman_smoothing"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_kalman_smoothing'].mean():.4f}")

def kalman_like_smoothing(history, op_sales_masked, outside_slice, alpha=0.2): # - Laura 
    imputed = op_sales_masked.copy()
    imputed_count = np.isnan(imputed).sum()

    n_hours = imputed.shape[1]

    for h in range(n_hours):
        s = pd.Series(imputed[:, h])

        # smoothing ähnlich wie einfacher State-Filter
        smooth = s.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()

        mask = s.isna()
        fallback = s.mean()

        imputed[mask.values, h] = smooth[mask].fillna(fallback).values

    imputed = np.maximum(imputed, 0)

    recovered_sum = np.nansum(imputed, axis=1)
    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_kalman_like"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Alpha used: {alpha}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_kalman_like'].mean():.4f}")

def stl_real(history, op_sales_masked, outside_slice, period=7): # Laura

    imputed = op_sales_masked.copy()
    imputed_count = np.isnan(imputed).sum()

    n_hours = imputed.shape[1]

    for h in range(n_hours):
        s = pd.Series(imputed[:, h]).astype(float)

        # STL kann keine NaNs direkt verarbeiten
        s_filled = s.interpolate(
            method="linear",
            limit_direction="both"
        )

        fallback = s.mean()
        s_filled = s_filled.fillna(fallback)

        try:
            stl = STL(s_filled, period=period, robust=True)

            result = stl.fit()

            estimate = result.trend + result.seasonal

            mask = s.isna()
            imputed[mask.values, h] = estimate[mask].values

        except Exception:
            mask = s.isna()
            imputed[mask.values, h] = fallback

    imputed = np.maximum(imputed, 0)

    recovered_sum = np.nansum(imputed, axis=1)
    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_stl_real"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"STL period used: {period}")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_stl_real'].mean():.4f}")

def stl_based(history, op_sales_masked, outside_slice, period=7):
    """
    STL-nahe Imputation:
    - nutzt Rolling Median als Trend
    - nutzt Wochentagsmuster als Saison
    - füllt NaNs mit Trend + Saison
    """

    imputed = op_sales_masked.copy()
    imputed_count = np.isnan(imputed).sum()

    weekdays = history["dt"].dt.weekday.values
    n_hours = imputed.shape[1]

    for h in range(n_hours):

        s = pd.Series(imputed[:, h])

        # Trend: geglätteter Verlauf
        trend = s.rolling(window=period, min_periods=1, center=True).median()

        # detrended Werte
        detrended = s - trend

        # Saison: durchschnittlicher Rest pro Wochentag
        seasonal = np.zeros(len(s))

        for wd in range(7):
            wd_mask = weekdays == wd
            seasonal_value = np.nanmean(detrended[wd_mask])

            if np.isnan(seasonal_value):
                seasonal_value = 0

            seasonal[wd_mask] = seasonal_value

        # Schätzung = Trend + Saison
        estimate = trend + seasonal

        # Fallback
        fallback = s.mean()
        estimate = estimate.fillna(fallback)

        mask = s.isna()
        imputed[mask.values, h] = estimate[mask].values

    imputed = np.maximum(imputed, 0)

    recovered_sum = np.nansum(imputed, axis=1)
    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_stl_based"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Period used: {period}")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_stl_based'].mean():.4f}")

# ML-basierte Recovery-Methoden: - Laura

def knn(history, op_sales_masked, outside_slice, n_neighbors=5):  # TODO Laura KNN-Imputation - Laura

    imputed = op_sales_masked.copy()

    imputed_count = np.isnan(imputed).sum()

    # KNNImputer arbeitet spaltenweise über die 16 Stunden
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")

    imputed = imputer.fit_transform(imputed)

    # Sicherheit: keine negativen Werte
    imputed = np.maximum(imputed, 0)

    # Rebuild corrected daily target
    recovered_sum = np.nansum(imputed, axis=1)

    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_knn"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"KNN neighbors used: {n_neighbors}")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_knn'].mean():.4f}")

def random_forest(history, op_sales_masked, outside_slice, max_train_rows=500_000, batch_size=500_000, random_state=42):  # Random Forest basierte Imputation - Laura

    print("\n=== Random Forest Recovery ===")

    start_total = time.time()

    imputed = op_sales_masked.copy()
    imputed_count = np.isnan(imputed).sum()

    print(f"Matrix shape: {imputed.shape}")
    print(f"Missing values: {imputed_count:,}")

    # ------------------------------------------------------------
    # 1. Trainingsdaten: nur sichtbare Stundenwerte
    # ------------------------------------------------------------

    rows_obs, hours_obs = np.where(~np.isnan(imputed)) # gibt alle Indizes der sichtbaren Stundenwerte zurück (rows_obs, hours_obs)


    X_obs = pd.DataFrame({
        "hour": hours_obs,
        "series_id": history["series_id"].values[rows_obs],
        "day_idx": history["day_idx"].values[rows_obs],
        "weekday": history["dt"].dt.weekday.values[rows_obs],
        "discount": history["discount"].values[rows_obs],
        "holiday_flag": history["holiday_flag"].values[rows_obs],
        "activity_flag": history["activity_flag"].values[rows_obs],
        "avg_temperature": history["avg_temperature"].fillna(0).values[rows_obs],
        "avg_humidity": history["avg_humidity"].fillna(0).values[rows_obs],
        "avg_wind_level": history["avg_wind_level"].fillna(0).values[rows_obs],
        "precpt": history["precpt"].fillna(0).values[rows_obs],
    })

    y_obs = imputed[rows_obs, hours_obs]

    print(f"Visible training rows: {len(X_obs):,}")

    # ------------------------------------------------------------
    # 2. Training sample ziehen, damit es schneller läuft
    # ------------------------------------------------------------

    if len(X_obs) > max_train_rows:
        sample_idx = np.random.default_rng(random_state).choice(
            len(X_obs),
            size=max_train_rows,
            replace=False
        )

        X_train = X_obs.iloc[sample_idx]
        y_train = y_obs[sample_idx]

    else:
        X_train = X_obs
        y_train = y_obs

    print(f"Training rows used: {len(X_train):,}")

    # ------------------------------------------------------------
    # 3. Modell trainieren
    # ------------------------------------------------------------

    model = RandomForestRegressor(n_estimators=50, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=random_state)

    print("Training Random Forest...")
    start_fit = time.time()

    model.fit(X_train, y_train)

    print(f"Training finished in {time.time() - start_fit:.2f} seconds")

    # ------------------------------------------------------------
    # 4. Fehlende Werte vorhersagen
    # ------------------------------------------------------------

    rows_miss, hours_miss = np.where(np.isnan(imputed))

    print(f"Predicting missing rows: {len(rows_miss):,}")

    start_pred = time.time()

    for start in range(0, len(rows_miss), batch_size):
        end = min(start + batch_size, len(rows_miss))

        print(f"Predicting batch {start:,} to {end:,}")

        batch_rows = rows_miss[start:end]
        batch_hours = hours_miss[start:end]

        X_missing = pd.DataFrame({
            "hour": batch_hours,
            "series_id": history["series_id"].values[batch_rows],
            "day_idx": history["day_idx"].values[batch_rows],
            "weekday": history["dt"].dt.weekday.values[batch_rows],
            "discount": history["discount"].values[batch_rows],
            "holiday_flag": history["holiday_flag"].values[batch_rows],
            "activity_flag": history["activity_flag"].values[batch_rows],
            "avg_temperature": history["avg_temperature"].fillna(0).values[batch_rows],
            "avg_humidity": history["avg_humidity"].fillna(0).values[batch_rows],
            "avg_wind_level": history["avg_wind_level"].fillna(0).values[batch_rows],
            "precpt": history["precpt"].fillna(0).values[batch_rows],
        })

        preds = model.predict(X_missing)

        imputed[batch_rows, batch_hours] = preds

    print(f"Prediction finished in {time.time() - start_pred:.2f} seconds")

    # ------------------------------------------------------------
    # 5. Rebuild corrected daily target
    # ------------------------------------------------------------

    imputed = np.maximum(imputed, 0)

    recovered_sum = np.nansum(imputed, axis=1)

    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_random_forest"] = recovered_daily

    print("\n=== Random Forest Recovery Finished ===")
    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_random_forest'].mean():.4f}")
    print(f"Total runtime: {time.time() - start_total:.2f} seconds")

def lightgbm(history, op_sales_masked, outside_slice, max_train_rows=500_000, batch_size=500_000, random_state=42):  # LightGBM-basierte Imputation - Laura

    print("\n=== LightGBM Recovery ===")

    start_total = time.time()

    imputed = op_sales_masked.copy()
    imputed_count = np.isnan(imputed).sum()

    print(f"Matrix shape: {imputed.shape}")
    print(f"Missing values: {imputed_count:,}")

    # ------------------------------------------------------------
    # 1. Trainingsdaten: nur sichtbare Stundenwerte
    # ------------------------------------------------------------

    rows_obs, hours_obs = np.where(~np.isnan(imputed))

    X_obs = pd.DataFrame({
        "hour": hours_obs,
        "series_id": history["series_id"].values[rows_obs],
        "day_idx": history["day_idx"].values[rows_obs],
        "weekday": history["dt"].dt.weekday.values[rows_obs],
        "discount": history["discount"].values[rows_obs],
        "holiday_flag": history["holiday_flag"].values[rows_obs],
        "activity_flag": history["activity_flag"].values[rows_obs],
        "avg_temperature": history["avg_temperature"].fillna(0).values[rows_obs],
        "avg_humidity": history["avg_humidity"].fillna(0).values[rows_obs],
        "avg_wind_level": history["avg_wind_level"].fillna(0).values[rows_obs],
        "precpt": history["precpt"].fillna(0).values[rows_obs],
    })

    y_obs = imputed[rows_obs, hours_obs]

    print(f"Visible training rows: {len(X_obs):,}")

    # ------------------------------------------------------------
    # 2. Sample ziehen, damit Training praktikabel bleibt
    # ------------------------------------------------------------

    if len(X_obs) > max_train_rows:
        sample_idx = np.random.default_rng(random_state).choice(
            len(X_obs),
            size=max_train_rows,
            replace=False
        )

        X_train = X_obs.iloc[sample_idx]
        y_train = y_obs[sample_idx]

    else:
        X_train = X_obs
        y_train = y_obs

    print(f"Training rows used: {len(X_train):,}")

    # ------------------------------------------------------------
    # 3. Modell trainieren
    # ------------------------------------------------------------

    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=-1, num_leaves=64, min_child_samples=50,
        subsample=0.8, colsample_bytree=0.8, objective="regression", n_jobs=-1, random_state=random_state, verbosity=-1)

    print("Training LightGBM...")
    start_fit = time.time()

    model.fit(X_train, y_train)

    print(f"Training finished in {time.time() - start_fit:.2f} seconds")

    # ------------------------------------------------------------
    # 4. Fehlende Werte vorhersagen
    # ------------------------------------------------------------

    rows_miss, hours_miss = np.where(np.isnan(imputed))

    print(f"Predicting missing rows: {len(rows_miss):,}")

    start_pred = time.time()

    for start in range(0, len(rows_miss), batch_size):
        end = min(start + batch_size, len(rows_miss))

        print(f"Predicting batch {start:,} to {end:,}")

        batch_rows = rows_miss[start:end]
        batch_hours = hours_miss[start:end]

        X_missing = pd.DataFrame({
            "hour": batch_hours,
            "series_id": history["series_id"].values[batch_rows],
            "day_idx": history["day_idx"].values[batch_rows],
            "weekday": history["dt"].dt.weekday.values[batch_rows],
            "discount": history["discount"].values[batch_rows],
            "holiday_flag": history["holiday_flag"].values[batch_rows],
            "activity_flag": history["activity_flag"].values[batch_rows],
            "avg_temperature": history["avg_temperature"].fillna(0).values[batch_rows],
            "avg_humidity": history["avg_humidity"].fillna(0).values[batch_rows],
            "avg_wind_level": history["avg_wind_level"].fillna(0).values[batch_rows],
            "precpt": history["precpt"].fillna(0).values[batch_rows],
        })

        preds = model.predict(X_missing)

        imputed[batch_rows, batch_hours] = preds

    print(f"Prediction finished in {time.time() - start_pred:.2f} seconds")

    # ------------------------------------------------------------
    # 5. Rebuild corrected daily target
    # ------------------------------------------------------------

    imputed = np.maximum(imputed, 0)

    recovered_sum = np.nansum(imputed, axis=1)

    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_lightgbm"] = recovered_daily

    print("\n=== LightGBM Recovery Finished ===")
    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_lightgbm'].mean():.4f}")
    print(f"Total runtime: {time.time() - start_total:.2f} seconds")

def lightgbm_v2(history, op_sales_masked, outside_slice,
                max_train_rows=1_500_000,
                batch_size=500_000,
                random_state=42):

    import time
    import numpy as np
    import pandas as pd
    import lightgbm as lgb

    print("\n=== LightGBM Recovery v2 ===")

    start_total = time.time()

    imputed = op_sales_masked.copy()
    imputed_count = np.isnan(imputed).sum()

    print(f"Matrix shape: {imputed.shape}")
    print(f"Missing values: {imputed_count:,}")

    # ------------------------------------------------------------
    # Kontext-Features pro Tageszeile
    # ------------------------------------------------------------
    dt = pd.to_datetime(history["dt"])

    visible_sum = np.nansum(op_sales_masked, axis=1)
    visible_count = np.sum(~np.isnan(op_sales_masked), axis=1)
    visible_mean = visible_sum / np.maximum(visible_count, 1)

    daily_raw = history["sale_amount"].values if "sale_amount" in history.columns else np.zeros(len(history))

    base = pd.DataFrame({
        "series_id": history["series_id"].values,
        "day_idx": history["day_idx"].values,
        "weekday": dt.dt.weekday.values,
        "month": dt.dt.month.values,
        "discount": history["discount"].values,
        "holiday_flag": history["holiday_flag"].values,
        "activity_flag": history["activity_flag"].values,
        "avg_temperature": history["avg_temperature"].fillna(0).values,
        "avg_humidity": history["avg_humidity"].fillna(0).values,
        "avg_wind_level": history["avg_wind_level"].fillna(0).values,
        "precpt": history["precpt"].fillna(0).values,
        "outside_slice": outside_slice,
        "visible_sum": visible_sum,
        "visible_count": visible_count,
        "visible_mean": visible_mean,
        "daily_raw": daily_raw,
    })

    optional_cols = [
        "product_id",
        "store_id",
        "city_id",
        "management_group_id",
        "psd",
    ]

    for col in optional_cols:
        if col in history.columns:
            base[col] = history[col].values

    base["weekday_sin"] = np.sin(2 * np.pi * base["weekday"] / 7)
    base["weekday_cos"] = np.cos(2 * np.pi * base["weekday"] / 7)
    base["month_sin"] = np.sin(2 * np.pi * base["month"] / 12)
    base["month_cos"] = np.cos(2 * np.pi * base["month"] / 12)

    # ------------------------------------------------------------
    # Helper: Feature-Matrix bauen
    # ------------------------------------------------------------
    def build_X(rows, hours):
        X = base.iloc[rows].reset_index(drop=True).copy()

        X["hour"] = hours
        X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
        X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)

        X["is_morning"] = ((X["hour"] >= 6) & (X["hour"] < 11)).astype(int)
        X["is_noon"] = ((X["hour"] >= 11) & (X["hour"] < 14)).astype(int)
        X["is_afternoon"] = ((X["hour"] >= 14) & (X["hour"] < 18)).astype(int)
        X["is_evening"] = ((X["hour"] >= 18) & (X["hour"] < 22)).astype(int)

        X["discount_x_hour"] = X["discount"] * X["hour"]
        X["holiday_x_hour"] = X["holiday_flag"] * X["hour"]
        X["activity_x_hour"] = X["activity_flag"] * X["hour"]

        return X

    # ------------------------------------------------------------
    # Trainingsdaten: nur sichtbare Stunden
    # ------------------------------------------------------------
    rows_obs, hours_obs = np.where(~np.isnan(imputed))
    y_obs = imputed[rows_obs, hours_obs]

    print(f"Visible training rows: {len(rows_obs):,}")

    if len(rows_obs) > max_train_rows:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(len(rows_obs), size=max_train_rows, replace=False)

        rows_train = rows_obs[sample_idx]
        hours_train = hours_obs[sample_idx]
        y_train = y_obs[sample_idx]
    else:
        rows_train = rows_obs
        hours_train = hours_obs
        y_train = y_obs

    X_train = build_X(rows_train, hours_train)

    print(f"Training rows used: {len(X_train):,}")

    # stärkere Gewichtung hoher Verkäufe
    sample_weight = np.sqrt(y_train + 1)

    # ------------------------------------------------------------
    # Modell
    # ------------------------------------------------------------
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        boosting_type="gbdt",

        n_estimators=700,
        learning_rate=0.035,

        num_leaves=127,
        max_depth=12,
        min_child_samples=40,

        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,

        reg_alpha=0.5,
        reg_lambda=1.5,

        n_jobs=-1,
        random_state=random_state,
        verbosity=-1
    )

    print("Training LightGBM v2...")
    start_fit = time.time()

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight
    )

    print(f"Training finished in {time.time() - start_fit:.2f} seconds")

    # ------------------------------------------------------------
    # Missing Values vorhersagen
    # ------------------------------------------------------------
    rows_miss, hours_miss = np.where(np.isnan(imputed))

    print(f"Predicting missing rows: {len(rows_miss):,}")

    start_pred = time.time()

    for start in range(0, len(rows_miss), batch_size):
        end = min(start + batch_size, len(rows_miss))

        print(f"Predicting batch {start:,} to {end:,}")

        batch_rows = rows_miss[start:end]
        batch_hours = hours_miss[start:end]

        X_missing = build_X(batch_rows, batch_hours)

        preds = model.predict(X_missing)
        preds = np.clip(preds, 0, None)

        imputed[batch_rows, batch_hours] = preds

    print(f"Prediction finished in {time.time() - start_pred:.2f} seconds")

    # ------------------------------------------------------------
    # Daily Sales rekonstruieren
    # ------------------------------------------------------------
    imputed = np.maximum(imputed, 0)

    recovered_sum = np.nansum(imputed, axis=1)
    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_lightgbm_v2"] = recovered_daily

    print("\n=== LightGBM Recovery v2 Finished ===")
    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_lightgbm_v2'].mean():.4f}")
    print(f"Total runtime: {time.time() - start_total:.2f} seconds")

    return recovered_daily

def xgboost(history, op_sales_masked, outside_slice, max_train_rows=500_000, batch_size=500_000, random_state=42):  # XGBoost-basierte Imputation - Laura

    print("\n=== XGBoost Recovery ===")

    start_total = time.time()

    imputed = op_sales_masked.copy()
    imputed_count = np.isnan(imputed).sum()

    print(f"Matrix shape: {imputed.shape}")
    print(f"Missing values: {imputed_count:,}")

    # ------------------------------------------------------------
    # 1. Trainingsdaten: nur sichtbare Stundenwerte
    # ------------------------------------------------------------

    rows_obs, hours_obs = np.where(~np.isnan(imputed))

    X_obs = pd.DataFrame({
        "hour": hours_obs,
        "series_id": history["series_id"].values[rows_obs],
        "day_idx": history["day_idx"].values[rows_obs],
        "weekday": history["dt"].dt.weekday.values[rows_obs],
        "discount": history["discount"].values[rows_obs],
        "holiday_flag": history["holiday_flag"].values[rows_obs],
        "activity_flag": history["activity_flag"].values[rows_obs],
        "avg_temperature": history["avg_temperature"].fillna(0).values[rows_obs],
        "avg_humidity": history["avg_humidity"].fillna(0).values[rows_obs],
        "avg_wind_level": history["avg_wind_level"].fillna(0).values[rows_obs],
        "precpt": history["precpt"].fillna(0).values[rows_obs],
    })

    y_obs = imputed[rows_obs, hours_obs]

    print(f"Visible training rows: {len(X_obs):,}")

    # ------------------------------------------------------------
    # 2. Sample ziehen, damit Training praktikabel bleibt
    # ------------------------------------------------------------

    if len(X_obs) > max_train_rows:
        sample_idx = np.random.default_rng(random_state).choice(
            len(X_obs),
            size=max_train_rows,
            replace=False
        )

        X_train = X_obs.iloc[sample_idx]
        y_train = y_obs[sample_idx]

    else:
        X_train = X_obs
        y_train = y_obs

    print(f"Training rows used: {len(X_train):,}")

    # ------------------------------------------------------------
    # 3. Modell trainieren
    # ------------------------------------------------------------

    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=8, min_child_weight=10, subsample=0.8,
        colsample_bytree=0.8, objective="reg:squarederror", n_jobs=-1, random_state=random_state, tree_method="hist")

    print("Training XGBoost...")
    start_fit = time.time()

    model.fit(X_train, y_train)

    print(f"Training finished in {time.time() - start_fit:.2f} seconds")

    # ------------------------------------------------------------
    # 4. Fehlende Werte vorhersagen
    # ------------------------------------------------------------

    rows_miss, hours_miss = np.where(np.isnan(imputed))

    print(f"Predicting missing rows: {len(rows_miss):,}")

    start_pred = time.time()

    for start in range(0, len(rows_miss), batch_size):
        end = min(start + batch_size, len(rows_miss))

        print(f"Predicting batch {start:,} to {end:,}")

        batch_rows = rows_miss[start:end]
        batch_hours = hours_miss[start:end]

        X_missing = pd.DataFrame({
            "hour": batch_hours,
            "series_id": history["series_id"].values[batch_rows],
            "day_idx": history["day_idx"].values[batch_rows],
            "weekday": history["dt"].dt.weekday.values[batch_rows],
            "discount": history["discount"].values[batch_rows],
            "holiday_flag": history["holiday_flag"].values[batch_rows],
            "activity_flag": history["activity_flag"].values[batch_rows],
            "avg_temperature": history["avg_temperature"].fillna(0).values[batch_rows],
            "avg_humidity": history["avg_humidity"].fillna(0).values[batch_rows],
            "avg_wind_level": history["avg_wind_level"].fillna(0).values[batch_rows],
            "precpt": history["precpt"].fillna(0).values[batch_rows],
        })

        preds = model.predict(X_missing)

        imputed[batch_rows, batch_hours] = preds

    print(f"Prediction finished in {time.time() - start_pred:.2f} seconds")

    # ------------------------------------------------------------
    # 5. Rebuild corrected daily target
    # ------------------------------------------------------------

    imputed = np.maximum(imputed, 0)

    recovered_sum = np.nansum(imputed, axis=1)

    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_xgboost"] = recovered_daily

    print("\n=== XGBoost Recovery Finished ===")
    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_xgboost'].mean():.4f}")
    print(f"Total runtime: {time.time() - start_total:.2f} seconds")

def iterative(history, op_sales_masked, outside_slice, max_iter=5, random_state=42):  # TODO Laura Iterative Imputation / MICE - Laura

    print("\n=== Iterative Imputation Recovery ===")

    start_total = time.time()

    imputed = op_sales_masked.copy()
    imputed_count = np.isnan(imputed).sum()

    print(f"Matrix shape: {imputed.shape}")
    print(f"Missing values: {imputed_count:,}")
    print(f"Max iterations: {max_iter}")

    estimator = ExtraTreesRegressor(n_estimators=30, max_depth=10, min_samples_leaf=20, n_jobs=-1, random_state=random_state)

    imputer = IterativeImputer(estimator=estimator, max_iter=max_iter, initial_strategy="mean", imputation_order="ascending", random_state=random_state, skip_complete=True, verbose=1)

    print("Starting iterative imputation...")
    start_impute = time.time()

    imputed = imputer.fit_transform(imputed)

    print(f"Iterative imputation finished in {time.time() - start_impute:.2f} seconds")

    imputed = np.maximum(imputed, 0)

    recovered_sum = np.nansum(imputed, axis=1)
    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_iterative"] = recovered_daily

    print("\n=== Iterative Imputation Recovery Finished ===")
    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_iterative'].mean():.4f}")
    print(f"Total runtime: {time.time() - start_total:.2f} seconds")

def iterative_improved(history, op_sales_masked, outside_slice, max_iter=5, random_state=42): #lädt über 5 Stunden dass hier: === Running recovery method: iterative_improved at 2026-06-12 14:42:30.694867 === === Improved Iterative Imputation Recovery === Starting iterative imputation... [IterativeImputer] Completing matrix with shape (4500000, 16) [IterativeImputer] Change: 14.01937198638916, scaled tolerance: 0.01690000109374523 [IterativeImputer] Change: 4.824539661407471, scaled tolerance: 0.01690000109374523 [IterativeImputer] Change: 2.6994433403015137, scaled tolerance: 0.01690000109374523
    import time
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import ExtraTreesRegressor

    print("\n=== Improved Iterative Imputation Recovery ===")
    start_total = time.time()

    imputed = op_sales_masked.copy()
    imputed_count = np.isnan(imputed).sum()

    # Erst grob mit Stundenmittel füllen als stabilerer Start
    hour_means = np.nanmean(imputed, axis=0)
    initial = np.where(np.isnan(imputed), hour_means, imputed)

    estimator = ExtraTreesRegressor(
        n_estimators=50,
        max_depth=12,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=random_state
    )

    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=max_iter,
        initial_strategy="mean",
        imputation_order="roman",
        random_state=random_state,
        skip_complete=True,
        verbose=1
    )

    print("Starting iterative imputation...")
    imputed_new = imputer.fit_transform(imputed)

    # Nur ursprüngliche NaNs ersetzen, sichtbare Werte behalten
    missing_mask = np.isnan(op_sales_masked)
    imputed[missing_mask] = imputed_new[missing_mask]

    imputed = np.maximum(imputed, 0)

    recovered_sum = np.nansum(imputed, axis=1)
    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_iterative_improved"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_iterative_improved'].mean():.4f}")
    print(f"Total runtime: {time.time() - start_total:.2f} seconds")

# Spezifische Demandrevovery Modelle: - Nils

def lost_sales_model(history): # was ist das? Laut ChatGPT ein Überbegriff für Modelle, die versuchen verlorene Umsätze zu schätzen, z.B. mit Random Forest oder XGBoost. 
    return

def tobit_model(history): # TODO
    # ---------- FEATURES ----------
    hours_matrix = np.vstack(history["hours_sale"].values)

    # Summarise the 24-hour vector into 4 interpretable features
    # instead of 24 raw columns — reduces parameters from 32 to 12,
    # much better identified on typical product-level sample sizes
    hours_stock = np.vstack(history["hours_stock_status"].values)  # (n, 24)
    peak_hours   = slice(9, 21)   # 09:00–20:00, main selling window

    hours_features = pd.DataFrame({
        "peak_sales":      hours_matrix[:, peak_hours].sum(axis=1),
        "offpeak_sales":   hours_matrix[:, :9].sum(axis=1) + hours_matrix[:, 21:].sum(axis=1),
        "peak_stockout_h": hours_stock[:, peak_hours].sum(axis=1),   # key censoring severity
        "avail_frac":      1 - history["stock_hour6_22_cnt"].values / 16,  # fraction of 6-22 available
    }, index=history.index)

    base_df = pd.DataFrame({
        "weekday":     pd.to_datetime(history["dt"]).dt.dayofweek,
        "temperature": history["avg_temperature"],
        "humidity":    history["avg_humidity"],
        "wind":        history["avg_wind_level"],
        "precpt":      history["precpt"],        # was missing — strong demand driver
        "holiday":     history["holiday_flag"],
        "activity":    history["activity_flag"],
        "discount":    history["discount"],
        "const":       1.0,
    }, index=history.index).fillna(0)

    X           = pd.concat([base_df, hours_features], axis=1).values.astype(np.float64)
    y           = history["sale_amount"].values.astype(np.float64).ravel()
    is_censored = history["is_censored"].values.astype(bool).ravel()

    obs_mask         = ~is_censored
    X_obs, y_obs     = X[obs_mask], y[obs_mask]
    X_cen            = X[is_censored]

    # Partial censoring weights: days that were out-of-stock for longer
    # get a stronger likelihood pull toward the censored branch
    avail_frac_cen = hours_features["avail_frac"].values[is_censored].clip(1e-3, 1 - 1e-3)

    _LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)

    def neg_log_likelihood(params):
        beta      = params[:-1]
        log_sigma = params[-1]
        sigma     = np.exp(log_sigma)

        mu_obs = X_obs @ beta
        mu_cen = X_cen @ beta

        z_obs  = (y_obs - mu_obs) / sigma
        ll_obs = -log_sigma - _LOG_SQRT_2PI - 0.5 * (z_obs * z_obs)

        # Weight censored log-likelihood by availability fraction:
        # a day with avail_frac=0.1 (mostly out-of-stock) contributes
        # more to the censored branch than one with avail_frac=0.9
        ll_cen = log_ndtr(-mu_cen / sigma) * (1 - avail_frac_cen)

        return -(ll_obs.sum() + ll_cen.sum())

    n_features = X.shape[1]
    result = minimize(neg_log_likelihood, np.zeros(n_features + 1, dtype=np.float64), method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-9},)

    beta_hat  = result.x[:-1]
    sigma_hat = float(np.exp(result.x[-1]))

    # ---------- PREDICT ----------
    mu_hat   = (X @ beta_hat).ravel()
    alpha    = mu_hat / sigma_hat
    pdf_a    = np.exp(-0.5 * alpha * alpha) / np.sqrt(2 * np.pi)
    cdf_a    = ndtr(alpha)
    lambda_  = pdf_a / np.maximum(cdf_a, 1e-12)

    e_y_star = mu_hat + sigma_hat * lambda_
    history["recovered_daily_sales_tobit"] = np.where(is_censored, np.maximum(e_y_star, 0), y)

    print(f"Converged: {result.success} | {result.message}")
    print(f"sigma_hat: {sigma_hat:.4f}")
    print(f"Mean raw sale_amount:  {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales:  {history['recovered_daily_sales_tobit'].mean():.4f}")

def tobit_model_improved(history):  # Tobit / censored regression for stockout recovery - Laura === Running recovery method: tobit_improved at 2026-06-12 19:46:54.791141 ===
# Converged: False | STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT
# sigma_hat: 0.1123
# Mean raw sale_amount:  0.9986
# Mean recovered sales:  0.5498
# Gespeichert: [0.  0.  5.3 ... 4.2 2.2 2.1]
# Verarbeitungszeit:  1:10:42.476935

    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize
    from scipy.special import log_ndtr
    from scipy.stats import norm
    from sklearn.preprocessing import StandardScaler

    print("\n=== Tobit Recovery Model ===")

    # ---------- FEATURES ----------
    hours_matrix = np.vstack(history["hours_sale"].values)
    hours_stock = np.vstack(history["hours_stock_status"].values)

    peak_hours = slice(9, 21)

    hours_features = pd.DataFrame({
        "peak_sales": hours_matrix[:, peak_hours].sum(axis=1),
        "offpeak_sales": hours_matrix[:, :9].sum(axis=1) + hours_matrix[:, 21:].sum(axis=1),
        "peak_stockout_h": hours_stock[:, peak_hours].sum(axis=1),
        "avail_frac": 1 - history["stock_hour6_22_cnt"].values / 16,
    }, index=history.index)

    base_df = pd.DataFrame({
        "weekday": pd.to_datetime(history["dt"]).dt.dayofweek,
        "temperature": history["avg_temperature"],
        "humidity": history["avg_humidity"],
        "wind": history["avg_wind_level"],
        "precpt": history["precpt"],
        "holiday": history["holiday_flag"],
        "activity": history["activity_flag"],
        "discount": history["discount"],
    }, index=history.index).fillna(0)

    X_df = pd.concat([base_df, hours_features], axis=1).fillna(0)

    y = history["sale_amount"].values.astype(np.float64).ravel()
    is_censored = history["is_censored"].values.astype(bool).ravel()

    # ---------- SCALE FEATURES ----------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values.astype(np.float64))

    # Intercept hinzufügen
    X = np.column_stack([np.ones(len(X_scaled)), X_scaled])

    obs_mask = ~is_censored
    cen_mask = is_censored

    X_obs = X[obs_mask]
    y_obs = y[obs_mask]

    X_cen = X[cen_mask]
    y_cen = y[cen_mask]

    avail_frac_cen = hours_features["avail_frac"].values[cen_mask].clip(1e-3, 1 - 1e-3)

    _LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)

    # ---------- NEGATIVE LOG LIKELIHOOD ----------
    def neg_log_likelihood(params):
        beta = params[:-1]
        log_sigma = params[-1]

        sigma = np.exp(log_sigma)

        mu_obs = X_obs @ beta
        mu_cen = X_cen @ beta

        # Uncensored observations:
        # y observed as normal outcome
        z_obs = (y_obs - mu_obs) / sigma

        ll_obs = (
            -log_sigma
            - _LOG_SQRT_2PI
            - 0.5 * (z_obs ** 2)
        )

        # Censored observations:
        # observed sales are a lower bound for true demand
        # P(Y >= y_cen) = 1 - Phi((y_cen - mu) / sigma)
        z_cen = (y_cen - mu_cen) / sigma

        ll_cen = log_ndtr(-z_cen)

        # stronger weighting when stockout severity is higher
        ll_cen = ll_cen * (1 - avail_frac_cen)

        return -(ll_obs.sum() + ll_cen.sum())

    # ---------- INITIALIZATION ----------
    n_features = X.shape[1]

    # OLS-like initialization on uncensored data
    beta_init, *_ = np.linalg.lstsq(X_obs, y_obs, rcond=None)

    residuals = y_obs - X_obs @ beta_init
    sigma_init = np.std(residuals)

    if sigma_init <= 0 or np.isnan(sigma_init):
        sigma_init = np.std(y_obs)

    if sigma_init <= 0 or np.isnan(sigma_init):
        sigma_init = 1.0

    params_init = np.append(beta_init, np.log(sigma_init))

    # ---------- OPTIMIZATION ----------
    result = minimize(
        neg_log_likelihood,
        params_init,
        method="L-BFGS-B",
        options={
            "maxiter": 3000,
            "ftol": 1e-7,
            "maxfun": 50000
        },
    )

    beta_hat = result.x[:-1]
    sigma_hat = float(np.exp(result.x[-1]))

    # ---------- PREDICT LATENT DEMAND ----------
    mu_hat = (X @ beta_hat).ravel()

    # Expected value conditional on Y >= observed y for censored rows
    z_all = (y - mu_hat) / sigma_hat

    pdf = norm.pdf(z_all)
    survival = np.maximum(1 - norm.cdf(z_all), 1e-12)

    lambda_ = pdf / survival

    expected_if_censored = mu_hat + sigma_hat * lambda_

    recovered = np.where(
        is_censored,
        np.maximum(expected_if_censored, y),
        y
    )

    history["recovered_daily_sales_tobit"] = recovered

    print(f"Converged: {result.success} | {result.message}")
    print(f"sigma_hat: {sigma_hat:.4f}")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_tobit'].mean():.4f}")

    if not result.success:
        print("Warning: Tobit optimization did not fully converge.")

def bayesian_model_old(history):  # Bayesisches Regressionsmodell mit Metropolis-Hastings MCMC # 7 min aber schlechter als raw_data
    # ---------- FEATURES (identisch zu Tobit) ----------
    hours_matrix = np.vstack(history["hours_sale"].values).astype(np.float32)
    hours_df = pd.DataFrame(hours_matrix, columns=[f"hour_{h}" for h in range(24)],
                            index=history.index)

    base_df = pd.DataFrame({
        "weekday":     pd.to_datetime(history["dt"]).dt.dayofweek.astype(np.float32),
        "temperature": history["avg_temperature"],
        "humidity":    history["avg_humidity"],
        "wind":        history["avg_wind_level"],
        "holiday":     history["holiday_flag"],
        "activity":    history["activity_flag"],
        "discount":    history["discount"],
        "const":       1.0,
    }, index=history.index).fillna(0)

    X           = pd.concat([base_df, hours_df], axis=1).values.astype(np.float32)
    y           = history["sale_amount"].values.astype(np.float32).ravel()
    is_censored = history["is_censored"].values.astype(bool).ravel()

    obs = ~is_censored
    cen =  is_censored
    X_obs, X_cen = X[obs], X[cen]
    y_obs        = y[obs]

    n_features = X.shape[1]

    # ---------- LOG POSTERIOR ----------
    # Prior: beta ~ N(0, 10²), log_sigma ~ N(0, 1)
    def log_posterior(params):
        beta      = params[:-1].astype(np.float32)
        log_sigma = params[-1]
        sigma     = float(np.exp(log_sigma))

        # --- log likelihood ---
        mu_obs  = (X_obs @ beta).ravel()
        r       = (y_obs - mu_obs) / sigma
        ll_obs  = -0.5 * (r ** 2).sum() - len(y_obs) * (np.log(sigma) + 0.5 * np.log(2 * np.pi))

        mu_cen  = (X_cen @ beta).ravel()
        alpha   = mu_cen / sigma
        ll_cen  = np.log(np.maximum(norm.cdf(alpha), 1e-12)).sum()

        ll = ll_obs + ll_cen

        # --- log prior ---
        lp_beta      = -0.5 * (beta ** 2 / 100).sum()   # N(0, 10²)
        lp_log_sigma = -0.5 * log_sigma ** 2             # N(0, 1)

        return float(ll + lp_beta + lp_log_sigma)

    # ---------- METROPOLIS-HASTINGS ----------
    n_samples    = 2000
    n_burnin     = 500
    step_size    = 0.01

    rng          = np.random.default_rng(42)
    params_curr  = np.zeros(n_features + 1, dtype=np.float64)
    lp_curr      = log_posterior(params_curr)

    samples      = np.empty((n_samples, n_features + 1), dtype=np.float32)
    n_accepted   = 0

    print(f"Running MCMC: {n_burnin} burnin + {n_samples} samples...")

    for i in range(n_burnin + n_samples):
        proposal = params_curr + rng.normal(0, step_size, size=params_curr.shape)
        lp_prop  = log_posterior(proposal)

        # accept/reject
        if np.log(rng.uniform()) < lp_prop - lp_curr:
            params_curr = proposal
            lp_curr     = lp_prop
            if i >= n_burnin:
                n_accepted += 1

        if i >= n_burnin:
            samples[i - n_burnin] = params_curr

    acceptance_rate = n_accepted / n_samples

    # ---------- POSTERIOR MEAN ESTIMATE ----------
    beta_hat  = samples[:, :-1].mean(axis=0).astype(np.float32)
    sigma_hat = float(np.exp(samples[:, -1].mean()))

    # ---------- PREDICT ----------
    mu_hat  = (X @ beta_hat).ravel()
    alpha   = mu_hat / sigma_hat
    lambda_ = norm.pdf(alpha) / np.maximum(norm.cdf(alpha), 1e-12)

    e_y_star  = mu_hat + sigma_hat * lambda_
    recovered = np.where(is_censored, np.maximum(e_y_star, 0), y)

    history["recovered_daily_sales_bayesian"] = recovered

    print(f"Acceptance rate: {acceptance_rate:.3f} (ideal: 0.2–0.5)")
    print(f"sigma_hat: {sigma_hat:.4f}")
    print(f"Mean raw sale_amount:  {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales:  {history['recovered_daily_sales_bayesian'].mean():.4f}")
    
def bayesian_model(history):  # Bayesisches Modell mit NUTS
    # ---------- FEATURES (identisch zu Tobit) ----------
    hours_matrix = np.vstack(history["hours_sale"].values).astype(np.float32)
    hours_df = pd.DataFrame(hours_matrix, columns=[f"hour_{h}" for h in range(24)],
                            index=history.index)

    base_df = pd.DataFrame({
        "weekday":     pd.to_datetime(history["dt"]).dt.dayofweek.astype(np.float32),
        "temperature": history["avg_temperature"],
        "humidity":    history["avg_humidity"],
        "wind":        history["avg_wind_level"],
        "holiday":     history["holiday_flag"],
        "activity":    history["activity_flag"],
        "discount":    history["discount"],
        "const":       1.0,
    }, index=history.index).fillna(0)

    X           = pd.concat([base_df, hours_df], axis=1).values.astype(np.float64)
    y           = history["sale_amount"].values.astype(np.float64).ravel()
    is_censored = history["is_censored"].values.astype(bool).ravel()

    obs = ~is_censored
    cen =  is_censored
    X_obs, X_cen = X[obs], X[cen]
    y_obs        = y[obs]
    n_features   = X.shape[1]

    # ---------- LOG POSTERIOR + GRADIENT ----------
    def log_posterior_and_grad(params):
        beta      = params[:-1]
        sigma     = float(np.exp(params[-1]))

        mu_obs    = (X_obs @ beta).ravel()
        r         = (y_obs - mu_obs) / sigma
        ll_obs    = -0.5 * (r ** 2).sum() - len(y_obs) * (np.log(sigma) + 0.5 * np.log(2 * np.pi))

        mu_cen    = (X_cen @ beta).ravel()
        alpha     = mu_cen / sigma
        phi       = norm.pdf(alpha)
        Phi       = np.maximum(norm.cdf(alpha), 1e-12)
        ll_cen    = np.log(Phi).sum()

        lp_beta      = -0.5 * (beta ** 2 / 100).sum()
        lp_log_sigma = -0.5 * params[-1] ** 2

        lp = ll_obs + ll_cen + lp_beta + lp_log_sigma

        # --- gradient ---
        grad_beta  =  X_obs.T @ (r / sigma)
        grad_beta +=  X_cen.T @ (phi / Phi / sigma)
        grad_beta -=  beta / 100                          # prior

        grad_log_sigma  =  (r ** 2).sum() - len(y_obs)   # uncensored
        grad_log_sigma -= (phi / Phi * alpha).sum()       # censored
        grad_log_sigma -=  params[-1]                     # prior

        return float(lp), np.append(grad_beta, grad_log_sigma)

    # ---------- NUTS SAMPLER ----------
    def nuts_step(params, log_grad_fn, step_size, max_depth=10):
        _, grad     = log_grad_fn(params)
        momentum    = np.random.randn(len(params))
        h_curr      = log_grad_fn(params)[0] - 0.5 * (momentum ** 2).sum()

        # leapfrog
        def leapfrog(q, p, eps, n_steps):
            q, p = q.copy(), p.copy()
            p   += 0.5 * eps * log_grad_fn(q)[1]
            for _ in range(n_steps - 1):
                q += eps * p
                p += eps * log_grad_fn(q)[1]
            q   += eps * p
            p   += 0.5 * eps * log_grad_fn(q)[1]
            return q, p

        # build tree
        q_minus = q_plus = params.copy()
        p_minus = p_plus = momentum.copy()
        q_prop  = params.copy()
        n, s    = 1, 1

        for depth in range(max_depth):
            direction = np.random.choice([-1, 1])
            eps       = direction * step_size
            n_steps   = 2 ** depth

            if direction == -1:
                q_minus, p_minus = leapfrog(q_minus, p_minus, eps, n_steps)
                q_new, p_new     = q_minus, p_minus
            else:
                q_plus,  p_plus  = leapfrog(q_plus,  p_plus,  eps, n_steps)
                q_new, p_new     = q_plus, p_plus

            lp_new = log_grad_fn(q_new)[0]
            h_new  = lp_new - 0.5 * (p_new ** 2).sum()

            # accept subtree proposal
            n_new  = int(np.exp(min(0.0, h_new - h_curr)))
            if n_new >= 1 and np.random.uniform() < n_new / n:
                q_prop = q_new.copy()
            n += n_new

            # U-turn check
            dq = q_plus - q_minus
            if (dq @ p_minus < 0) or (dq @ p_plus < 0):
                break

        return q_prop

    # ---------- WARMUP: dual averaging step size ----------
    def dual_averaging_warmup(params, log_grad_fn, n_warmup=200, target_accept=0.65):
        step_size   = 0.1
        mu          = np.log(10 * step_size)
        log_eps_bar = 0.0
        h_bar       = 0.0
        gamma, t0, kappa = 0.05, 10, 0.75

        for t in range(1, n_warmup + 1):
            params   = nuts_step(params, log_grad_fn, step_size)
            lp, grad = log_grad_fn(params)
            h_bar    = (1 - 1/(t + t0)) * h_bar + (target_accept - min(1, np.exp(lp))) / (t + t0)
            log_eps  = mu - np.sqrt(t) / gamma * h_bar
            step_size        = np.exp(log_eps)
            log_eps_bar      = t**(-kappa) * log_eps + (1 - t**(-kappa)) * log_eps_bar

        return params, np.exp(log_eps_bar)

    # ---------- RUN ----------
    n_samples = 1000
    n_warmup  = 300

    params0  = np.zeros(n_features + 1)

    # MAP initialisierung für besseren Startpunkt
    map_result = minimize(lambda p: -log_posterior_and_grad(p)[0], params0, jac=lambda p: -log_posterior_and_grad(p)[1], method="L-BFGS-B",)
    params_curr = map_result.x
    print("MAP initialization done, starting NUTS warmup...")

    params_curr, step_size = dual_averaging_warmup(params_curr, log_posterior_and_grad, n_warmup)
    print(f"Warmup done — adapted step_size: {step_size:.5f}, starting sampling...")

    samples = np.empty((n_samples, n_features + 1))
    for i in range(n_samples):
        params_curr  = nuts_step(params_curr, log_posterior_and_grad, step_size)
        samples[i]   = params_curr
        if (i + 1) % 100 == 0:
            print(f"  Sample {i+1}/{n_samples}")

    # ---------- POSTERIOR MEAN ----------
    beta_hat  = samples[:, :-1].mean(axis=0)
    sigma_hat = float(np.exp(samples[:, -1].mean()))

    # ---------- PREDICT ----------
    mu_hat   = (X @ beta_hat).ravel()
    alpha    = mu_hat / sigma_hat
    lambda_  = norm.pdf(alpha) / np.maximum(norm.cdf(alpha), 1e-12)
    e_y_star = mu_hat + sigma_hat * lambda_
    recovered = np.where(is_censored, np.maximum(e_y_star, 0), y)

    history["recovered_daily_sales_bayesian"] = recovered

    print(f"sigma_hat: {sigma_hat:.4f}")
    print(f"Mean raw sale_amount:  {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales:  {history['recovered_daily_sales_bayesian'].mean():.4f}")

def inventory_aware_model(history): # inventory aware model is a forecasting method
    return


# Deep Learning basierte Recovery-Methoden: - Nils 

def autoencoder(history, op_sales_masked, outside_slice, latent_dim=8, epochs=20, lr=1e-3, batch_size=256, device=None):

    # Architecture: input = [16 sales + 16 mask flags + 7 covariates] = 39-dim
    # Encoder: 39 → 64 → latent_dim
    # Decoder: latent_dim → 64 → 16  (reconstructs all hours)
    # Loss: MSE on observed hours only (same censored-loss idea as transformer)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    N, H = op_sales_masked.shape

    # ── Covariates (same as transformer version) ──────────────────────────────
    def norm(x): return (x - x.mean()) / (x.std() + 1e-8)

    cov = np.column_stack([
        norm(history["discount"].values),
        history["holiday_flag"].values,
        history["activity_flag"].values,
        norm(history["avg_temperature"].values),
        norm(history["avg_humidity"].values),
        norm(history["precpt"].values),
    ]).astype(np.float32)                          # (N, 6)

    # ── Normalise sales ───────────────────────────────────────────────────────
    observed = op_sales_masked[~np.isnan(op_sales_masked)]
    sale_mean, sale_std = observed.mean(), observed.std() + 1e-8
    sales_norm = (op_sales_masked - sale_mean) / sale_std
    obs_mask   = (~np.isnan(sales_norm)).astype(np.float32)   # 1=observed, 0=censored
    sales_input = np.nan_to_num(sales_norm, nan=0.0).astype(np.float32)

    # Input: sales (16) + obs_mask (16) + covariates (6) = 38
    X    = np.concatenate([sales_input, obs_mask, cov], axis=1)   # (N, 38)
    tgt  = sales_norm.copy().astype(np.float32)                    # (N, 16) — NaN in censored

    T_X    = torch.tensor(X)
    T_tgt  = torch.tensor(tgt)
    T_obs  = torch.tensor(obs_mask, dtype=torch.bool)

    loader = DataLoader(TensorDataset(T_X, T_tgt, T_obs),
                        batch_size=batch_size, shuffle=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    class DemandAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(38, 64), nn.GELU(),
                nn.Linear(64, 32), nn.GELU(),
                nn.Linear(32, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 32), nn.GELU(),
                nn.Linear(32, 64),         nn.GELU(),
                nn.Linear(64, H),          # reconstruct all 16 hours
            )
        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = DemandAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"Training autoencoder on {device}  |  params: {sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_n = 0.0, 0
        for x, tgt_b, obs_b in loader:
            x, tgt_b, obs_b = x.to(device), tgt_b.to(device), obs_b.to(device)
            pred = model(x)                              # (B, 16)
            loss = nn.functional.huber_loss(pred[obs_b], tgt_b[obs_b], delta=1.0)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * obs_b.sum().item()
            total_n    += obs_b.sum().item()
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{epochs}  loss={total_loss/max(total_n,1):.5f}")

    # ── Inference ─────────────────────────────────────────────────────────────
    model.eval()
    preds = []
    with torch.no_grad():
        for (x,) in DataLoader(TensorDataset(T_X), batch_size=batch_size):
            preds.append(model(x.to(device)).cpu().numpy())
    preds_denorm = (np.concatenate(preds) * sale_std + sale_mean).clip(0)

    imputed = op_sales_masked.copy()
    nan_mask = np.isnan(imputed)
    imputed_count = nan_mask.sum()
    imputed[nan_mask] = preds_denorm[nan_mask]

    # ── Rebuild daily (same as global_mean / transformer) ─────────────────────
    recovered_sum   = np.nansum(imputed, axis=1)
    recovered_daily = outside_slice + recovered_sum
    history["recovered_daily_sales_autoencoder"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount:      {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales:      {history['recovered_daily_sales_autoencoder'].mean():.4f}")

#def transformer(history): # SAITS, BRITS, GRIN, CSDI

def transformer_old(history, op_sales_masked, outside_slice, epochs=20, batch_size=1024, random_state=42):  # TODO Transformer-basierte Imputation - Laura

    print("\n=== Transformer Recovery ===")
    start_total = time.time()

    torch.manual_seed(random_state)

    imputed = op_sales_masked.copy()
    imputed_count = np.isnan(imputed).sum()

    # ------------------------------------------------------------
    # 1. Trainingsdaten: nur vollständig sichtbare Profile
    # ------------------------------------------------------------

    clean_mask = ~np.isnan(imputed).any(axis=1)
    clean_profiles = imputed[clean_mask]

    print(f"Clean training profiles: {len(clean_profiles):,}")
    print(f"Missing values to impute: {imputed_count:,}")

    # ------------------------------------------------------------
    # 2. Skalieren
    # ------------------------------------------------------------

    # Keep rows that have at least one non-NaN value for training
    clean_mask = ~np.isnan(imputed).all(axis=1)
    clean_profiles = imputed[clean_mask]

    clean_profiles_filled = np.where(np.isnan(clean_profiles), np.nanmean(clean_profiles, axis=0), clean_profiles)

    X_train, y_train = train_test_split(clean_profiles_filled, test_size=0.1, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    y_train_scaled = scaler.transform(y_train)

    dataset = TensorDataset(X_train_scaled, y_train_scaled)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ------------------------------------------------------------
    # 3. Einfaches Transformer Autoencoder Modell
    # ------------------------------------------------------------

    class Transformer(nn.Module):
        def __init__(self, seq_len=24, d_model=32, nhead=4, num_layers=2):
            super().__init__()

            self.input_proj = nn.Linear(1, d_model)

            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True)

            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            self.output_proj = nn.Linear(d_model, 1)

        def forward(self, x):
            # x shape: (batch, 24)
            x = x.unsqueeze(-1)          # (batch, 24, 1)
            x = self.input_proj(x)       # (batch, 24, d_model)
            x = self.encoder(x)          # (batch, 24, d_model)
            x = self.output_proj(x)      # (batch, 24, 1)
            return x.squeeze(-1)         # (batch, 24)

    model = Transformer(seq_len=imputed.shape[1])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # ------------------------------------------------------------
    # 4. Training
    # ------------------------------------------------------------

    print("Training Transformer...")

    for epoch in range(epochs):
        epoch_loss = 0

        for xb, yb in loader:
            optimizer.zero_grad()

            pred = model(xb)

            loss = loss_fn(pred, yb)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / len(loader):.6f}")

    # ------------------------------------------------------------
    # 5. Fehlende Stunden rekonstruieren
    # ------------------------------------------------------------

    print("Imputing missing values...")

    hour_mean = scaler.mean_

    for start in range(0, len(imputed), batch_size):
        end = min(start + batch_size, len(imputed))

        batch = imputed[start:end]

        missing_mask = np.isnan(batch)

        if not missing_mask.any():
            continue

        # NaNs vor Modellinput mit Stundenmittel füllen
        batch_filled = np.where(missing_mask, hour_mean, batch)

        batch_scaled = scaler.transform(batch_filled)

        xb = torch.tensor(batch_scaled, dtype=torch.float32)

        with torch.no_grad():
            reconstructed_scaled = model(xb).numpy()

        reconstructed = scaler.inverse_transform(reconstructed_scaled)

        batch[missing_mask] = reconstructed[missing_mask]

        imputed[start:end] = batch

    # ------------------------------------------------------------
    # 6. Rebuild corrected daily target
    # ------------------------------------------------------------

    imputed = np.maximum(imputed, 0)

    recovered_sum = np.nansum(imputed, axis=1)

    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_transformer"] = recovered_daily

    print("\n=== Transformer Recovery Finished ===")
    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Epochs used: {epochs}")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_transformer'].mean():.4f}")
    print(f"Total runtime: {time.time() - start_total:.2f} seconds")

def transformer_old2(history, op_sales_masked, outside_slice, epochs=20, batch_size=1024, random_state=42):

    print("\n=== Transformer Recovery ===")
    start_total = time.time()

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    imputed = op_sales_masked.copy()
    imputed_count = int(np.isnan(imputed).sum())
    seq_len = imputed.shape[1]

    # ------------------------------------------------------------
    # 1. Trainingsdaten: nur vollständig sichtbare Profile
    # ------------------------------------------------------------

    clean_mask = ~np.isnan(imputed).any(axis=1)   # rows with NO missing values
    clean_profiles = imputed[clean_mask]            # shape: (n_clean, seq_len)

    print(f"Clean training profiles: {len(clean_profiles):,}")
    print(f"Missing values to impute: {imputed_count:,}")

    if len(clean_profiles) == 0:
        raise ValueError("No fully clean profiles found — cannot train.")

    # ------------------------------------------------------------
    # 2. Skalieren
    # ------------------------------------------------------------

    X_train, X_val = train_test_split(clean_profiles, test_size=0.1, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit only on train split
    X_val_scaled   = scaler.transform(X_val)

    hour_mean_scaled = np.zeros(seq_len)             # mean in scaled space = 0 by definition

    # ------------------------------------------------------------
    # 3. Masked Autoencoding Dataset
    #    Input:  clean profile with ~30 % positions zeroed out (= scaled mean)
    #    Target: original clean profile
    #    → model learns to recover masked positions from context
    # ------------------------------------------------------------

    def make_corrupted(X_scaled, mask_ratio=0.3):
        corrupted = X_scaled.copy()
        mask = np.random.rand(*corrupted.shape) < mask_ratio   # True = position is masked
        corrupted[mask] = 0.0                                  # 0 = scaled mean
        return corrupted.astype(np.float32), mask.astype(np.float32)

    X_train_corrupted, train_masks = make_corrupted(X_train_scaled)
    X_val_corrupted,   val_masks   = make_corrupted(X_val_scaled)

    train_dataset = TensorDataset(
        torch.tensor(X_train_corrupted),
        torch.tensor(X_train_scaled.astype(np.float32)),
        torch.tensor(train_masks),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_corrupted),
        torch.tensor(X_val_scaled.astype(np.float32)),
        torch.tensor(val_masks),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------
    # 4. Transformer
    # ------------------------------------------------------------

    class Transformer(nn.Module):
        def __init__(self, seq_len, d_model=64, nhead=4, num_layers=2, dropout=0.1):
            super().__init__()
            self.input_proj  = nn.Linear(1, d_model)
            encoder_layer    = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=128, dropout=dropout,
                batch_first=True
            )
            self.encoder     = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_proj = nn.Linear(d_model, 1)

        def forward(self, x):
            # x: (batch, seq_len)
            x = x.unsqueeze(-1)       # (batch, seq_len, 1)
            x = self.input_proj(x)    # (batch, seq_len, d_model)
            x = self.encoder(x)       # (batch, seq_len, d_model)
            x = self.output_proj(x)   # (batch, seq_len, 1)
            return x.squeeze(-1)      # (batch, seq_len)

    model     = Transformer(seq_len=seq_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    loss_fn   = nn.MSELoss(reduction="none")   # element-wise so we can mask

    # ------------------------------------------------------------
    # 5. Training  (loss only on corrupted positions)
    # ------------------------------------------------------------

    print("Training Transformer...")

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(epochs):

        # --- train ---
        model.train()
        train_loss = 0.0
        for xb, yb, mb in train_loader:
            optimizer.zero_grad()
            pred        = model(xb)                        # (batch, seq_len)
            elem_loss   = loss_fn(pred, yb)                # (batch, seq_len)
            masked_loss = (elem_loss * mb).sum() / mb.sum().clamp(min=1)
            masked_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += masked_loss.item()

        train_loss /= len(train_loader)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                pred        = model(xb)
                elem_loss   = loss_fn(pred, yb)
                masked_loss = (elem_loss * mb).sum() / mb.sum().clamp(min=1)
                val_loss   += masked_loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1:>3}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

    # restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model (val_loss={best_val_loss:.6f})")

    # ------------------------------------------------------------
    # 6. Fehlende Stunden rekonstruieren
    # ------------------------------------------------------------

    print("Imputing missing values...")
    model.eval()

    for start in range(0, len(imputed), batch_size):
        end   = min(start + batch_size, len(imputed))
        batch = imputed[start:end].copy()

        missing_mask = np.isnan(batch)

        if not missing_mask.any():
            continue

        # fill NaNs with per-hour mean before scaling
        batch_filled = np.where(missing_mask, scaler.mean_, batch)
        batch_scaled = scaler.transform(batch_filled).astype(np.float32)

        xb = torch.tensor(batch_scaled)

        with torch.no_grad():
            reconstructed_scaled = model(xb).numpy()

        reconstructed = scaler.inverse_transform(reconstructed_scaled)

        # write back ONLY the originally missing positions
        batch[missing_mask] = reconstructed[missing_mask]
        imputed[start:end]  = batch

    # ------------------------------------------------------------
    # 7. Rebuild corrected daily target
    # ------------------------------------------------------------

    imputed          = np.maximum(imputed, 0)
    recovered_sum    = np.nansum(imputed, axis=1)
    recovered_daily  = outside_slice + recovered_sum

    history["recovered_daily_sales_transformer"] = recovered_daily

    print("\n=== Transformer Recovery Finished ===")
    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Epochs used: {epochs}")
    print(f"Mean raw sale_amount:    {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales:    {history['recovered_daily_sales_transformer'].mean():.4f}")
    print(f"Total runtime: {time.time() - start_total:.2f} seconds")

def transformer_old3(history, op_sales_masked, outside_slice, epochs=20, batch_size=512, random_state=42):
    """
    Denoising Transformer for hourly sales imputation.

    op_sales_masked : np.ndarray, shape (n_days, n_hours)
                      Hourly sales profiles; NaN = censored hour.
    outside_slice   : np.ndarray, shape (n_days,)
                      Sales recorded outside the tracked hour window.
    history         : pd.DataFrame
                      Full dataset; result written to 'recovered_daily_sales_transformer'.

    Approach — BERT-style masked prediction:
      1. Train only on fully observed profiles.
      2. During training: randomly mask ~30 % of hours (replace with learned [MASK] token).
         Loss computed only on masked positions → model learns to infer missing hours
         from the surrounding hourly context.
      3. At inference: replace NaN positions with [MASK], reconstruct, write back only
         the originally missing positions.
    """

    print("\n=== Transformer Recovery ===")
    start_total = time.time()

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    n_days, n_hours = op_sales_masked.shape

    # ------------------------------------------------------------------ #
    # 1.  Collect fully observed profiles for training                    #
    # ------------------------------------------------------------------ #

    fully_observed = ~np.isnan(op_sales_masked).any(axis=1)
    clean = op_sales_masked[fully_observed].astype(np.float32)   # (n_clean, n_hours)

    print(f"Fully observed profiles : {len(clean):,} / {n_days:,}")
    print(f"Missing hourly cells    : {int(np.isnan(op_sales_masked).sum()):,}")

    if len(clean) == 0:
        raise ValueError("No fully observed profiles — cannot train.")

    # ------------------------------------------------------------------ #
    # 2.  Scale (fit on clean profiles)                                   #
    # ------------------------------------------------------------------ #

    scaler = StandardScaler()
    clean_scaled = scaler.fit_transform(clean).astype(np.float32)   # (n_clean, n_hours)

    # ------------------------------------------------------------------ #
    # 3.  Model                                                           #
    # ------------------------------------------------------------------ #

    class HourlySalesTransformer(nn.Module):
        """
        BERT-style Transformer for hourly profile imputation.

        Each hour is treated as one token.  A learnable [MASK] embedding
        replaces censored positions.  Positional encodings give the model
        awareness of time-of-day structure.
        """

        def __init__(self, n_hours, d_model=64, nhead=4, num_layers=3, dropout=0.1):
            super().__init__()

            self.n_hours = n_hours
            self.d_model = d_model

            # project scalar sales value → d_model
            self.value_proj = nn.Linear(1, d_model)

            # learnable mask token (replaces censored hours)
            self.mask_token = nn.Parameter(torch.zeros(d_model))

            # learnable positional encoding (one per hour)
            self.pos_emb = nn.Embedding(n_hours, d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,        # Pre-LN: more stable training
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )

            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )

        def forward(self, x, mask):
            """
            x    : (batch, n_hours)   — scaled sales values; arbitrary at masked positions
            mask : (batch, n_hours)   — 1.0 = position is masked (to be predicted)
            """
            batch = x.size(0)
            pos   = torch.arange(self.n_hours, device=x.device)   # (n_hours,)

            # project observed values
            tokens = self.value_proj(x.unsqueeze(-1))              # (batch, n_hours, d_model)

            # replace masked positions with learned mask token
            mask_expanded = mask.unsqueeze(-1).bool()              # (batch, n_hours, 1)
            mask_tok       = self.mask_token.view(1, 1, -1).expand(batch, self.n_hours, -1)
            tokens         = torch.where(mask_expanded, mask_tok, tokens)

            # add positional encoding
            tokens = tokens + self.pos_emb(pos).unsqueeze(0)       # (batch, n_hours, d_model)

            # transformer
            out = self.encoder(tokens)                             # (batch, n_hours, d_model)

            # project to scalar
            return self.head(out).squeeze(-1)                      # (batch, n_hours)

    model = HourlySalesTransformer(n_hours=n_hours).to("cpu")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters        : {total_params:,}")

    # ------------------------------------------------------------------ #
    # 4.  Training                                                        #
    # ------------------------------------------------------------------ #

    X_train, X_val = train_test_split(clean_scaled, test_size=0.1, random_state=random_state)

    X_train_t = torch.tensor(X_train)
    X_val_t   = torch.tensor(X_val)

    train_ds = TensorDataset(X_train_t)
    val_ds   = TensorDataset(X_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    MASK_RATIO = 0.30

    def masked_loss(pred, target, mask):
        """MSE only on masked positions."""
        err = (pred - target) ** 2
        denom = mask.sum().clamp(min=1)
        return (err * mask).sum() / denom

    best_val  = float("inf")
    best_state = None

    print(f"\nTraining for up to {epochs} epochs  (mask ratio={MASK_RATIO:.0%})")
    print(f"{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'LR':>10}")
    print("-" * 42)

    for epoch in range(1, epochs + 1):

        # --- train ---
        model.train()
        tr_loss = 0.0
        for (xb,) in train_loader:
            rand_mask = (torch.rand_like(xb) < MASK_RATIO).float()
            optimizer.zero_grad()
            pred = model(xb, rand_mask)
            loss = masked_loss(pred, xb, rand_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        # --- validate ---
        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for (xb,) in val_loader:
                rand_mask = (torch.rand_like(xb) < MASK_RATIO).float()
                pred      = model(xb, rand_mask)
                vl_loss  += masked_loss(pred, xb, rand_mask).item()
        vl_loss /= len(val_loader)

        scheduler.step(vl_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:>6}  {tr_loss:>10.6f}  {vl_loss:>10.6f}  {lr:>10.2e}")

        if vl_loss < best_val:
            best_val   = vl_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"\nRestored best checkpoint  (val={best_val:.6f})")

    # ------------------------------------------------------------------ #
    # 5.  Imputation                                                      #
    # ------------------------------------------------------------------ #

    print("Imputing missing values...")
    model.eval()

    imputed = op_sales_masked.copy().astype(np.float64)

    for start in range(0, n_days, batch_size):
        end   = min(start + batch_size, n_days)
        batch = imputed[start:end]                          # (b, n_hours)

        nan_mask = np.isnan(batch)
        if not nan_mask.any():
            continue

        # fill NaN positions with per-hour mean before scaling
        batch_filled = np.where(nan_mask, scaler.mean_, batch).astype(np.float32)
        batch_scaled = scaler.transform(batch_filled).astype(np.float32)

        xb   = torch.tensor(batch_scaled)
        mb   = torch.tensor(nan_mask.astype(np.float32))   # 1 = was NaN → predict this

        with torch.no_grad():
            recon_scaled = model(xb, mb).numpy()            # (b, n_hours)

        recon = scaler.inverse_transform(recon_scaled)

        # write back only originally missing positions
        batch[nan_mask] = recon[nan_mask]
        imputed[start:end] = batch

    # ------------------------------------------------------------------ #
    # 6.  Rebuild daily totals                                            #
    # ------------------------------------------------------------------ #

    imputed = np.maximum(imputed, 0)                        # no negative sales
    recovered_daily = outside_slice + imputed.sum(axis=1)

    history["recovered_daily_sales_transformer"] = recovered_daily

    print("\n=== Transformer Recovery Finished ===")
    print(f"Imputed hourly cells    : {int(np.isnan(op_sales_masked).sum()):,}")
    print(f"Mean raw sale_amount    : {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales    : {history['recovered_daily_sales_transformer'].mean():.4f}")
    print(f"Total runtime           : {time.time() - start_total:.2f}s")

def transformer(history, op_sales_masked, outside_slice, d_model=32, n_heads=4, n_layers=2, d_ff=64, epochs=20, lr=3e-4, batch_size=256, device=None):
    """
    Imputes censored (NaN) hourly cells in op_sales_masked using an
    encoder-only Transformer trained only on observed (non-NaN) hours.

    Mirrors the signature of global_mean():
      - history         : DataFrame with covariates + sale_amount
      - op_sales_masked : (N, 16) float32 array, NaN where stockout
      - outside_slice   : (N,)    float32 array, sales outside h06-h21

    Writes history["recovered_daily_sales_transformer"] and returns imputed.
    """
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    N, H = op_sales_masked.shape   # (N_rows, 16 hours)

    # ── 1. Build covariates matrix  (N, H, C) ────────────────────────────────
    # Broadcast daily scalars to every hour slot so each token is self-contained
    hour_idx   = np.tile(np.arange(H), (N, 1)).astype(np.float32) / (H - 1)  # (N,16) normalised 0-1
    discount   = np.repeat(history["discount"].values[:, None],          H, axis=1).astype(np.float32)
    holiday    = np.repeat(history["holiday_flag"].values[:, None],      H, axis=1).astype(np.float32)
    activity   = np.repeat(history["activity_flag"].values[:, None],     H, axis=1).astype(np.float32)
    temperature= np.repeat(history["avg_temperature"].values[:, None],   H, axis=1).astype(np.float32)
    humidity   = np.repeat(history["avg_humidity"].values[:, None],      H, axis=1).astype(np.float32)
    precpt     = np.repeat(history["precpt"].values[:, None],            H, axis=1).astype(np.float32)

    # Normalise continuous covariates
    def norm(x):
        mu, sigma = x.mean(), x.std() + 1e-8
        return (x - mu) / sigma

    covariates = np.stack([
        hour_idx,
        norm(discount),
        holiday,
        activity,
        norm(temperature),
        norm(humidity),
        norm(precpt),
    ], axis=-1)                                      # (N, 16, 7)
    C = covariates.shape[-1]

    # ── 2. Normalise sales on observed hours only ─────────────────────────────
    observed_vals = op_sales_masked[~np.isnan(op_sales_masked)]
    sale_mean = observed_vals.mean()
    sale_std  = observed_vals.std() + 1e-8

    sales_norm = (op_sales_masked - sale_mean) / sale_std    # NaN preserved
    obs_mask   = ~np.isnan(sales_norm)                       # True = observed

    # Replace NaN with 0 for tensor input (model sees mask, not NaN)
    sales_input = np.nan_to_num(sales_norm, nan=0.0).astype(np.float32)

    # ── 3. Tensors ────────────────────────────────────────────────────────────
    T_sales = torch.tensor(sales_input,    dtype=torch.float32)   # (N, H)
    T_cov   = torch.tensor(covariates,     dtype=torch.float32)   # (N, H, C)
    T_obs   = torch.tensor(obs_mask,       dtype=torch.bool)      # (N, H)
    T_tgt   = torch.tensor(sales_norm.copy().astype(np.float32))  # (N, H)  NaN in stockout

    dataset = TensorDataset(T_sales, T_cov, T_obs, T_tgt)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ── 4. Model ──────────────────────────────────────────────────────────────
    class HourlyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(1 + C, d_model)   # sale + covariates → d_model

            # Fixed sinusoidal positional encoding
            pe = torch.zeros(H, d_model)
            pos = torch.arange(H).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))   # (1, H, d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=0.1, batch_first=True, activation="gelu"
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Linear(d_model, 1)

        def forward(self, sale, cov):
            # sale: (B, H)   cov: (B, H, C)
            x = torch.cat([sale.unsqueeze(-1), cov], dim=-1)  # (B, H, 1+C)
            x = self.input_proj(x) + self.pe                  # (B, H, d_model)
            x = self.encoder(x)                               # (B, H, d_model)
            return self.head(x).squeeze(-1)                   # (B, H)

    model = HourlyTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print(f"Training transformer on {device}  |  params: {sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_tokens = 0.0, 0
        for sale, cov, obs, tgt in loader:
            sale, cov, obs, tgt = sale.to(device), cov.to(device), obs.to(device), tgt.to(device)
            pred = model(sale, cov)             # (B, H)

            # ── Censored loss: only backprop through observed hours ──────────
            if obs.sum() == 0:
                continue
            loss = nn.functional.huber_loss(pred[obs], tgt[obs], delta=1.0)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss   += loss.item() * obs.sum().item()
            epoch_tokens += obs.sum().item()

        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{epochs}  loss={epoch_loss/max(epoch_tokens,1):.5f}")

    # ── 6. Inference: fill NaN cells ─────────────────────────────────────────
    model.eval()
    all_preds = []
    inf_loader = DataLoader(TensorDataset(T_sales, T_cov), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for sale, cov in inf_loader:
            pred = model(sale.to(device), cov.to(device))
            all_preds.append(pred.cpu().numpy())

    preds_norm = np.concatenate(all_preds, axis=0)                    # (N, H)
    preds_denorm = (preds_norm * sale_std + sale_mean).clip(0)        # (N, H)

    # Only replace NaN (censored) cells; keep observed sales unchanged
    imputed = op_sales_masked.copy()
    nan_mask = np.isnan(imputed)
    imputed_count = nan_mask.sum()
    imputed[nan_mask] = preds_denorm[nan_mask]

    # ── 7. Rebuild daily totals (same as global_mean) ─────────────────────────
    recovered_sum   = np.nansum(imputed, axis=1)
    recovered_daily = outside_slice + recovered_sum
    history["recovered_daily_sales_transformer"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount:     {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales:     {history['recovered_daily_sales_transformer'].mean():.4f}")


def diffusion_model(history, op_sales_masked, outside_slice, noise_scale=0.1, n_samples=5, random_state=42):  # Diffusion-like Recovery - Laura

    print("\n=== Diffusion-like Recovery ===")

    start_total = time.time()

    rng = np.random.default_rng(random_state)

    imputed = op_sales_masked.copy()

    imputed_count = np.isnan(imputed).sum()

    print(f"Matrix shape: {imputed.shape}")
    print(f"Missing values: {imputed_count:,}")
    print(f"Number of samples: {n_samples}")
    print(f"Noise scale: {noise_scale}")

    # ------------------------------------------------------------
    # 1. Stundenmittel und Stundenstandardabweichung berechnen
    # ------------------------------------------------------------

    hour_mean = np.nanmean(imputed, axis=0)
    hour_std = np.nanstd(imputed, axis=0)

    # Falls std 0 oder NaN ist
    hour_std = np.where(np.isnan(hour_std) | (hour_std == 0), 1e-6, hour_std)

    nan_mask = np.isnan(imputed)

    # ------------------------------------------------------------
    # 2. Mehrere plausible Werte sampeln
    # ------------------------------------------------------------

    sampled_values = []

    for i in range(n_samples):

        noise = rng.normal(loc=0, scale=noise_scale, size=imputed.shape)

        sample = hour_mean + noise * hour_std

        sample = np.maximum(sample, 0)

        sampled_values.append(sample)

    # Durchschnitt der Samples
    generated = np.mean(sampled_values, axis=0)

    # Nur fehlende Werte ersetzen
    imputed[nan_mask] = generated[nan_mask]

    # ------------------------------------------------------------
    # 3. Rebuild corrected daily target
    # ------------------------------------------------------------

    recovered_sum = np.nansum(imputed, axis=1)

    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_diffusion"] = recovered_daily

    print("\n=== Diffusion-like Recovery Finished ===")
    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_diffusion'].mean():.4f}")
    print(f"Total runtime: {time.time() - start_total:.2f} seconds")


# TODO 
# DLinear (Nils)

class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend component.
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=0
        )

    def forward(self, x):
        # x: (batch, seq_len)

        pad = (self.kernel_size - 1) // 2

        front = x[:, 0:1].repeat(1, pad)
        end = x[:, -1:].repeat(1, pad)

        x = torch.cat([front, x, end], dim=1)

        x = self.avg(x.unsqueeze(1)).squeeze(1)

        return x

class SeriesDecomp(nn.Module):

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):

        trend = self.moving_avg(x)

        seasonal = x - trend

        return seasonal, trend

class DLinear(nn.Module):

    def __init__(self, seq_len=16, individual=False, kernel_size=5):
        super().__init__()

        self.seq_len = seq_len
        self.individual = individual

        self.decomposition = SeriesDecomp(kernel_size)

        if individual:

            self.linear_seasonal = nn.ModuleList([nn.Linear(seq_len, seq_len)])

            self.linear_trend = nn.ModuleList([nn.Linear(seq_len, seq_len)])

        else:

            self.linear_seasonal = nn.Linear(seq_len, seq_len)

            self.linear_trend = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        """
        x shape = (batch, 16)
        """

        seasonal, trend = self.decomposition(x)

        if self.individual:

            seasonal = self.linear_seasonal[0](seasonal)
            trend = self.linear_trend[0](trend)

        else:

            seasonal = self.linear_seasonal(seasonal)
            trend = self.linear_trend(trend)

        out = seasonal + trend

        return out
    
class RecoveryDataset(Dataset):
    """
    Creates training samples for demand recovery.

    Input  : hourly sales with randomly masked hours
    Target : original hourly sales

    Parameters
    ----------
    hourly_sales : ndarray (n_days, 16)
        Complete hourly sales (no NaNs).
    mask_prob : float
        Probability that an observed hour is hidden.
    """

    def __init__(self, hourly_sales, mask_prob=0.30, keep_min_hours=8,):

        self.hourly_sales = hourly_sales.astype(np.float32)
        self.mask_prob = mask_prob
        self.keep_min_hours = keep_min_hours

    def __len__(self):
        return len(self.hourly_sales)

    def __getitem__(self, idx):

        target = self.hourly_sales[idx].copy()

        x = target.copy()

        # Random mask
        mask = np.random.rand(len(x)) < self.mask_prob

        # Keep at least some observations
        if mask.sum() > len(x) - self.keep_min_hours:

            keep = np.random.choice(len(x), self.keep_min_hours, replace=False)

            mask[:] = True
            mask[keep] = False

        # Artificial stockout
        x[mask] = 0.0

        return (torch.from_numpy(x), torch.from_numpy(target))
    

def dlinear_train(history, op_sales, op_sales_masked, epochs=50, batch_size=256, lr=1e-3, mask_prob=0.30, device=None, random_state=42):
    print(op_sales_masked.shape) # TODO
    hourly_sales = op_sales.copy()

    # Keep only complete days
    complete_days = ~np.isnan(hourly_sales).any(axis=1)

    train_data = hourly_sales[complete_days]
        
    train_dataset = RecoveryDataset(train_data, mask_prob=0.30)

    # TODO delete if it works
    x, y = train_dataset[0]
    print("Input")
    print(x)
    print("Target")
    print(y)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Keep only complete days
    complete_days = ~np.isnan(op_sales_masked).any(axis=1)

    train_data = op_sales_masked[complete_days]

    print(f"Training days: {len(train_data):,}")

    dataset = RecoveryDataset(train_data, mask_prob=mask_prob)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False,)

    # Model
    seq_len = hourly_sales.shape[1]
    model = DLinear(seq_len=seq_len)

    model.to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    model.train()

    for epoch in range(epochs):

        running_loss = 0.0

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            pred = model(x)

            loss = criterion(pred, y)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * len(x)

        epoch_loss = running_loss / len(dataset)

        print(f"Epoch {epoch+1:3d}/{epochs} | Loss = {epoch_loss:.5f}")

    # Save model
    model_path="dlinear_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to '{model_path}'")

    return model

def dlinear(history, op_sales, op_sales_masked, outside_slice):
    """
    Recover hourly demand using a pretrained DLinear model.
    """

    # ---------------- LOAD TRAINED MODEL ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DLinear()
    if os.path.exists("dlinear_model.pt"):
        model.load_state_dict(torch.load("dlinear_model.pt", map_location=device))
    else:
        model = dlinear_train(history, op_sales, op_sales_masked)

    model.load_state_dict(torch.load("dlinear_model.pt", map_location=device))
    model.to(device)
    model.eval()

    # ---------------- PREPARE INPUT ----------------
    imputed = op_sales_masked.copy()

    nan_mask = np.isnan(imputed)

    # DLinear cannot handle NaNs
    model_input = np.nan_to_num(imputed, nan=0.0)

    x = torch.tensor(model_input, dtype=torch.float32, device=device)

    # shape: (batch, 16)
    x = x.unsqueeze(1)

    # ---------------- PREDICT ----------------
    with torch.no_grad():
        prediction = model(x).squeeze(1).cpu().numpy()

    # ---------------- IMPUTE ----------------
    imputed[nan_mask] = prediction[nan_mask]
    imputed = np.maximum(imputed, 0)

    # ---------------- DAILY RECOVERY ----------------
    recovered_sum = np.sum(imputed, axis=1)

    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_dlinear"] = recovered_daily

    print(f"Imputed {nan_mask.sum():,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_dlinear'].mean():.4f}")

    return imputed
