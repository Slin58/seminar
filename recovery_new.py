# recovery.py contains all recovery methods we implemented
# recovery methods need to add column: "recovered_daily_sales" to train

import numpy as np
import pandas as pd
import os
import time
import lightgbm as lgb
from xgboost import XGBRegressor

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

from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import norm

# Einfache Imputation Methoden:
def random_sampling(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val, rng):
    imputed_train = op_sales_masked_train.copy()
    imputed_val = op_sales_masked_val.copy()
    n_hours = imputed_train.shape[1]

    for h in range(n_hours):
        pool = imputed_train[~np.isnan(imputed_train[:, h]), h]  # sampling pool: train only

        mask_train = np.isnan(imputed_train[:, h])
        if mask_train.sum() > 0:
            imputed_train[mask_train, h] = np.maximum(0, rng.choice(pool, size=mask_train.sum(), replace=True))

        mask_val = np.isnan(imputed_val[:, h])
        if mask_val.sum() > 0:
            imputed_val[mask_val, h] = np.maximum(0, rng.choice(pool, size=mask_val.sum(), replace=True))

    recovered_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    return pd.concat([
        pd.Series(recovered_train, index=train.index, name="recovered_daily_sales_random_sampling"),
        pd.Series(recovered_val, index=val.index, name="recovered_daily_sales_random_sampling"),
    ]).sort_index()

def global_mean(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val):  # globaler Durchschnitt
    imputed_train = op_sales_masked_train.copy()
    imputed_val = op_sales_masked_val.copy()

    # Durchschnitt über alle train Stundenwerte
    mean_value = np.nanmean(imputed_train)
    
    # NaN-Werte ersetzen bei train
    mask_train = np.isnan(imputed_train)
    imputed_count_train = mask_train.sum()
    imputed_train[mask_train] = mean_value

    # NaN-Werte ersetzen bei val
    mask_val = np.isnan(imputed_val)
    imputed_count_val = mask_val.sum()
    imputed_val[mask_val] = mean_value

    # Rebuild corrected daily target
    recovered_sum_train = np.nansum(imputed_train, axis=1)
    recovered_daily_train = outside_slice_train + recovered_sum_train
    recovered_sum_val = np.nansum(imputed_val, axis=1)
    recovered_daily_val = outside_slice_val + recovered_sum_val

    print(f"recovered_daily_train_mean: {np.nanmean(recovered_daily_train):.4f}")
    print(f"recovered_daily_val_mean: {np.nanmean(recovered_daily_val):.4f}")

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_global_mean")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_global_mean")

    history_r = pd.concat([recovered_train_series, recovered_val_series ]).sort_index() # Series 0: Tag 1, 2, 3 - 83 Series n: Tag 1, 2, 3 - 83; series 0 84-90

    print(f"Imputed {imputed_count_train:,} hourly cells")
    print(f"Imputed {imputed_count_val:,} hourly cells")
    print(f"Global mean used: {mean_value:.4f}")
    return history_r

def series_mean(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val):
    imputed_train = op_sales_masked_train.copy()
    imputed_val = op_sales_masked_val.copy()

    codes_train = train["series_id"].values.astype(int)
    codes_val = val["series_id"].values.astype(int)
    n_series = codes_train.max() + 1

    global_hourly_means = np.nanmean(imputed_train, axis=0)  # globaler Fallback pro Stunde (nur train)

    valid_train = ~np.isnan(imputed_train)
    filled_train = np.where(valid_train, imputed_train, 0.0)

    sums = pd.DataFrame(filled_train).groupby(codes_train).sum().reindex(range(n_series), fill_value=0).values
    counts = pd.DataFrame(valid_train.astype(float)).groupby(codes_train).sum().reindex(range(n_series), fill_value=0).values

    with np.errstate(invalid="ignore"):
        series_hourly_means = np.where(counts > 0, sums / np.where(counts > 0, counts, 1), global_hourly_means)

    mask_train = np.isnan(imputed_train)
    imputed_count_train = mask_train.sum()
    imputed_train[mask_train] = series_hourly_means[codes_train][mask_train]

    mask_val = np.isnan(imputed_val)
    imputed_count_val = mask_val.sum()
    imputed_val[mask_val] = series_hourly_means[codes_val][mask_val]

    recovered_daily_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_daily_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    print(f"Imputed {imputed_count_train:,} hourly cells (train)")
    print(f"Imputed {imputed_count_val:,} hourly cells (val)")
    print(f"recovered_daily_train_mean: {np.nanmean(recovered_daily_train):.4f}")
    print(f"recovered_daily_val_mean: {np.nanmean(recovered_daily_val):.4f}")

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_series_mean")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_series_mean")

    return pd.concat([recovered_train_series, recovered_val_series]).sort_index()

def series_daily_mean(train, val):  # Durchschnitt derselben series_id pro Tag - Nils
    recovered_daily_train = train["sale_amount"].where(train["is_censored"] == 0, np.nan)
    recovered_daily_val = val["sale_amount"].where(val["is_censored"] == 0, np.nan)

    codes_train = train["series_id"].values.astype(int)
    codes_val = val["series_id"].values.astype(int)
    n_series = codes_train.max() + 1

    global_mean = recovered_daily_train.mean()  # globaler Fallback (nur train)

    grouped = recovered_daily_train.groupby(codes_train)
    sums = grouped.sum().reindex(range(n_series), fill_value=0).values
    counts = grouped.count().reindex(range(n_series), fill_value=0).values

    series_mean = np.where(counts > 0, sums / np.where(counts > 0, counts, 1), global_mean)

    recovered_daily_train = recovered_daily_train.fillna(pd.Series(series_mean[codes_train], index=train.index))
    recovered_daily_val = recovered_daily_val.fillna(pd.Series(series_mean[codes_val], index=val.index))

    recovered_daily_train.name = "recovered_daily_sales_series_daily_mean"
    recovered_daily_val.name = "recovered_daily_sales_series_daily_mean"

    return pd.concat([recovered_daily_train, recovered_daily_val]).sort_index()

def weekday_mean(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val):
    imputed_train = op_sales_masked_train.copy()
    imputed_val = op_sales_masked_val.copy()
    n_hours = imputed_train.shape[1]

    wd_train = pd.to_datetime(train["dt"]).dt.weekday.values
    wd_val = pd.to_datetime(val["dt"]).dt.weekday.values

    valid = ~np.isnan(imputed_train)
    filled = np.where(valid, imputed_train, 0.0)
    sums = np.stack([np.bincount(wd_train, weights=filled[:, h], minlength=7) for h in range(n_hours)], axis=1)
    counts = np.stack([np.bincount(wd_train, weights=valid[:, h].astype(float), minlength=7) for h in range(n_hours)], axis=1)

    if counts.sum() == 0:
        weekday_means = np.nanmean(imputed_train, axis=0) # globaler Fallback
    else:
        weekday_means = sums / counts

    for imputed, wd in [(imputed_train, wd_train), (imputed_val, wd_val)]:
        mask = np.isnan(imputed)
        imputed[mask] = weekday_means[wd][mask]
        np.maximum(imputed, 0, out=imputed)

    recovered_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    return pd.concat([
        pd.Series(recovered_train, index=train.index, name="recovered_daily_sales_weekday_mean"),
        pd.Series(recovered_val, index=val.index, name="recovered_daily_sales_weekday_mean"),
    ]).sort_index()

def weekday_daily_mean(train, val):  # Durchschnitt desselben Wochentags - Nils
    recovered_daily_train = train["sale_amount"].where(train["is_censored"] == 0, np.nan)
    recovered_daily_val = val["sale_amount"].where(val["is_censored"] == 0, np.nan)

    dayofweek_train = pd.to_datetime(train["dt"]).dt.dayofweek
    dayofweek_val = pd.to_datetime(val["dt"]).dt.dayofweek

    global_mean = recovered_daily_train.mean()  # globaler Fallback (nur train)

    # Durchschnitt pro Wochentag (nur train)
    grouped = recovered_daily_train.groupby(dayofweek_train)
    sums = grouped.sum().reindex(range(7), fill_value=0)
    counts = grouped.count().reindex(range(7), fill_value=0)

    weekday_mean = np.where(counts > 0, sums / counts.replace(0, 1), global_mean)
    weekday_mean = pd.Series(weekday_mean, index=range(7))

    recovered_daily_train = recovered_daily_train.fillna(dayofweek_train.map(weekday_mean))
    recovered_daily_val = recovered_daily_val.fillna(dayofweek_val.map(weekday_mean))

    recovered_daily_train.index = train.index
    recovered_daily_val.index = val.index

    recovered_daily_train.name = "recovered_daily_sales_weekday_daily_mean"
    recovered_daily_val.name = "recovered_daily_sales_weekday_daily_mean"

    return pd.concat([recovered_daily_train, recovered_daily_val]).sort_index()

def hourly_mean(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val):
    imputed_train = op_sales_masked_train.copy()
    imputed_val = op_sales_masked_val.copy()

    hourly_means = np.nanmean(imputed_train, axis=0)  # fit on train only

    for imputed in (imputed_train, imputed_val):
        mask = np.isnan(imputed)
        imputed[mask] = np.broadcast_to(hourly_means, imputed.shape)[mask]
        np.maximum(imputed, 0, out=imputed)

    recovered_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    return pd.concat([
        pd.Series(recovered_train, index=train.index, name="recovered_daily_sales_hourly_mean"),
        pd.Series(recovered_val, index=val.index, name="recovered_daily_sales_hourly_mean"),
    ]).sort_index()

# Moving averages: - Laura
def rolling_mean(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val, window=7):
    imputed_train = op_sales_masked_train.copy()
    imputed_val = op_sales_masked_val.copy()
    n_hours = imputed_train.shape[1]

    global_means = np.nanmean(imputed_train, axis=0)  # fallback, train only

    for h in range(n_hours):
        s_train = pd.Series(imputed_train[:, h])
        roll_train = s_train.rolling(window, min_periods=1).mean()
        mask_train = s_train.isna()
        imputed_train[mask_train.values, h] = roll_train[mask_train].fillna(global_means[h]).values

        s_val = pd.Series(imputed_val[:, h])
        roll_val = s_val.rolling(window, min_periods=1).mean()
        mask_val = s_val.isna()
        imputed_val[mask_val.values, h] = roll_val[mask_val].fillna(global_means[h]).values

    recovered_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    return pd.concat([
        pd.Series(recovered_train, index=train.index, name="recovered_daily_sales_rolling_mean"),
        pd.Series(recovered_val, index=val.index, name="recovered_daily_sales_rolling_mean"),
    ]).sort_index()

def exponential_moving_average(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val, alpha=0.3):  # EMA - Laura
    imputed_train = op_sales_masked_train.copy()
    imputed_val = op_sales_masked_val.copy()
    n_hours = imputed_train.shape[1]

    global_hourly_means = np.nanmean(imputed_train, axis=0)  # globaler Fallback pro Stunde (nur train)

    for h in range(n_hours):
        s_train = pd.Series(imputed_train[:, h])
        ema_train = s_train.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
        mask_train = s_train.isna()
        imputed_train[mask_train.values, h] = ema_train[mask_train].fillna(global_hourly_means[h]).values

        s_val = pd.Series(imputed_val[:, h])
        ema_val = s_val.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
        mask_val = s_val.isna()
        imputed_val[mask_val.values, h] = ema_val[mask_val].fillna(global_hourly_means[h]).values

    imputed_count_train = np.isnan(op_sales_masked_train).sum()
    imputed_count_val = np.isnan(op_sales_masked_val).sum()

    recovered_daily_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_daily_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    print(f"Imputed {imputed_count_train:,} hourly cells (train)")
    print(f"Imputed {imputed_count_val:,} hourly cells (val)")
    print(f"Alpha used: {alpha}")

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_exponential_moving_average")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_exponential_moving_average")

    return pd.concat([recovered_train_series, recovered_val_series]).sort_index()

def exponential_moving_average_series(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val, alpha=0.3):
    imputed_train = op_sales_masked_train.copy()
    imputed_val = op_sales_masked_val.copy()
    n_hours = imputed_train.shape[1]

    codes_train = train["series_id"].values.astype(int)
    codes_val = val["series_id"].values.astype(int)

    global_hourly_means = np.nanmean(imputed_train, axis=0)  # globaler Fallback pro Stunde (nur train)

    for h in range(n_hours):
        s_train = pd.Series(imputed_train[:, h])
        ema_train = s_train.groupby(codes_train).transform(
            lambda x: x.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
        )
        mask_train = s_train.isna()
        imputed_train[mask_train.values, h] = ema_train[mask_train].fillna(global_hourly_means[h]).values

        s_val = pd.Series(imputed_val[:, h])
        ema_val = s_val.groupby(codes_val).transform(
            lambda x: x.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
        )
        mask_val = s_val.isna()
        imputed_val[mask_val.values, h] = ema_val[mask_val].fillna(global_hourly_means[h]).values

    imputed_train = np.maximum(imputed_train, 0)
    imputed_val = np.maximum(imputed_val, 0)

    imputed_count_train = np.isnan(op_sales_masked_train).sum()
    imputed_count_val = np.isnan(op_sales_masked_val).sum()

    recovered_daily_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_daily_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    print(f"Imputed {imputed_count_train:,} hourly cells (train)")
    print(f"Imputed {imputed_count_val:,} hourly cells (val)")
    print(f"Alpha used: {alpha}")

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_exponential_moving_average_series")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_exponential_moving_average_series")

    return pd.concat([recovered_train_series, recovered_val_series]).sort_index()

# Zeitreihenbasierte Recovery-Methoden: - Laura

def interpolation_linear(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val):  # Lineare Interpolation - Laura
    imputed_train = op_sales_masked_train.copy()
    imputed_val = op_sales_masked_val.copy()
    n_hours = imputed_train.shape[1]

    global_hourly_means = np.nanmean(imputed_train, axis=0)  # globaler Fallback pro Stunde (nur train)

    for h in range(n_hours):
        s_train = pd.Series(imputed_train[:, h]).interpolate(method="linear", limit_direction="both")
        imputed_train[:, h] = s_train.fillna(global_hourly_means[h]).values

        s_val = pd.Series(imputed_val[:, h]).interpolate(method="linear", limit_direction="both")
        imputed_val[:, h] = s_val.fillna(global_hourly_means[h]).values

    imputed_train = np.maximum(imputed_train, 0)
    imputed_val = np.maximum(imputed_val, 0)

    imputed_count_train = np.isnan(op_sales_masked_train).sum()
    imputed_count_val = np.isnan(op_sales_masked_val).sum()

    recovered_daily_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_daily_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    print(f"Imputed {imputed_count_train:,} hourly cells (train)")
    print(f"Imputed {imputed_count_val:,} hourly cells (val)")

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_interpolation_linear")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_interpolation_linear")

    return pd.concat([recovered_train_series, recovered_val_series]).sort_index()

def interpolation_spline(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val, order=3):  # Spline-Interpolation - Laura
    imputed_train = op_sales_masked_train.copy()
    imputed_val = op_sales_masked_val.copy()
    n_hours = imputed_train.shape[1]

    global_hourly_means = np.nanmean(imputed_train, axis=0)  # globaler Fallback pro Stunde (nur train)

    for h in range(n_hours):
        s_train = pd.Series(imputed_train[:, h])
        interp_train = s_train.interpolate(method="spline", order=order, limit_direction="both") \
            if s_train.notna().sum() > order else s_train.copy()
        imputed_train[:, h] = interp_train.fillna(global_hourly_means[h]).values

        s_val = pd.Series(imputed_val[:, h])
        interp_val = s_val.interpolate(method="spline", order=order, limit_direction="both") \
            if s_val.notna().sum() > order else s_val.copy()
        imputed_val[:, h] = interp_val.fillna(global_hourly_means[h]).values

    imputed_train = np.maximum(imputed_train, 0)
    imputed_val = np.maximum(imputed_val, 0)

    imputed_count_train = np.isnan(op_sales_masked_train).sum()
    imputed_count_val = np.isnan(op_sales_masked_val).sum()

    recovered_daily_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_daily_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    print(f"Imputed {imputed_count_train:,} hourly cells (train)")
    print(f"Imputed {imputed_count_val:,} hourly cells (val)")
    print(f"Spline order used: {order}")

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_interpolation_spline")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_interpolation_spline")

    return pd.concat([recovered_train_series, recovered_val_series]).sort_index()

def interpolation_spline_series(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val, order=3):
    imputed_train = op_sales_masked_train.copy()
    imputed_val = op_sales_masked_val.copy()
    n_hours = imputed_train.shape[1]

    codes_train = train["series_id"].values.astype(int)
    codes_val = val["series_id"].values.astype(int)

    global_hourly_means = np.nanmean(imputed_train, axis=0)  # globaler Fallback pro Stunde (nur train)

    def interp_within_series(s, codes):
        return s.groupby(codes).transform(
            lambda x: x.interpolate(method="spline", order=order, limit_direction="both")
            if x.notna().sum() > order else x
        )

    for h in range(n_hours):
        s_train = pd.Series(imputed_train[:, h])
        interp_train = interp_within_series(s_train, codes_train)
        imputed_train[:, h] = interp_train.fillna(global_hourly_means[h]).values

        s_val = pd.Series(imputed_val[:, h])
        interp_val = interp_within_series(s_val, codes_val)
        imputed_val[:, h] = interp_val.fillna(global_hourly_means[h]).values

    imputed_train = np.maximum(imputed_train, 0)
    imputed_val = np.maximum(imputed_val, 0)

    imputed_count_train = np.isnan(op_sales_masked_train).sum()
    imputed_count_val = np.isnan(op_sales_masked_val).sum()

    recovered_daily_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_daily_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    print(f"Imputed {imputed_count_train:,} hourly cells (train)")
    print(f"Imputed {imputed_count_val:,} hourly cells (val)")
    print(f"Spline order used: {order}")

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_interpolation_spline_series")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_interpolation_spline_series")

    return pd.concat([recovered_train_series, recovered_val_series]).sort_index()

def interpolation_polynomial(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val, order=2):  # Polynomial-Interpolation - Laura
    imputed_train = op_sales_masked_train.copy()
    imputed_val = op_sales_masked_val.copy()
    n_hours = imputed_train.shape[1]

    global_hourly_means = np.nanmean(imputed_train, axis=0)  # globaler Fallback pro Stunde (nur train)

    for h in range(n_hours):
        s_train = pd.Series(imputed_train[:, h])
        interp_train = s_train.interpolate(method="polynomial", order=order, limit_direction="both") \
            if s_train.notna().sum() > order else s_train.copy()
        imputed_train[:, h] = interp_train.fillna(global_hourly_means[h]).values

        s_val = pd.Series(imputed_val[:, h])
        interp_val = s_val.interpolate(method="polynomial", order=order, limit_direction="both") \
            if s_val.notna().sum() > order else s_val.copy()
        imputed_val[:, h] = interp_val.fillna(global_hourly_means[h]).values

    imputed_train = np.maximum(imputed_train, 0)
    imputed_val = np.maximum(imputed_val, 0)

    imputed_count_train = np.isnan(op_sales_masked_train).sum()
    imputed_count_val = np.isnan(op_sales_masked_val).sum()

    recovered_daily_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_daily_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    print(f"Imputed {imputed_count_train:,} hourly cells (train)")
    print(f"Imputed {imputed_count_val:,} hourly cells (val)")
    print(f"Polynomial order used: {order}")

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_interpolation_polynomial")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_interpolation_polynomial")

    return pd.concat([recovered_train_series, recovered_val_series]).sort_index()

def kalman_smoothing(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val):  # Kalman Smoothing / State Space - Laura
    imputed_train = op_sales_masked_train.copy()
    imputed_val = op_sales_masked_val.copy()
    n_hours = imputed_train.shape[1]

    global_hourly_means = np.nanmean(imputed_train, axis=0)  # globaler Fallback pro Stunde (nur train)

    def smooth_column(col, fallback):
        s = pd.Series(col).astype(float)

        if s.notna().sum() < 10:
            return s.fillna(fallback).values

        try:
            model = UnobservedComponents(s, level="local level")
            result = model.fit(disp=False)
            smoothed = result.smoothed_state[0]

            filled = s.copy()
            filled[s.isna()] = smoothed[s.isna()]
            filled = filled.fillna(fallback)
            return filled.values
        except Exception:
            return s.fillna(fallback).values

    for h in range(n_hours):
        imputed_train[:, h] = smooth_column(imputed_train[:, h], global_hourly_means[h])
        imputed_val[:, h] = smooth_column(imputed_val[:, h], global_hourly_means[h])

    imputed_train = np.maximum(imputed_train, 0)
    imputed_val = np.maximum(imputed_val, 0)

    imputed_count_train = np.isnan(op_sales_masked_train).sum()
    imputed_count_val = np.isnan(op_sales_masked_val).sum()

    recovered_daily_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_daily_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    print(f"Imputed {imputed_count_train:,} hourly cells (train)")
    print(f"Imputed {imputed_count_val:,} hourly cells (val)")

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_kalman_smoothing")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_kalman_smoothing")

    return pd.concat([recovered_train_series, recovered_val_series]).sort_index()


def kalman_like_smoothing(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val, alpha=0.2):  # - Laura
    imputed_train = op_sales_masked_train.copy()
    imputed_val = op_sales_masked_val.copy()
    n_hours = imputed_train.shape[1]

    global_hourly_means = np.nanmean(imputed_train, axis=0)  # globaler Fallback pro Stunde (nur train)

    for h in range(n_hours):
        s_train = pd.Series(imputed_train[:, h])
        smooth_train = s_train.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
        mask_train = s_train.isna()
        imputed_train[mask_train.values, h] = smooth_train[mask_train].fillna(global_hourly_means[h]).values

        s_val = pd.Series(imputed_val[:, h])
        smooth_val = s_val.ewm(alpha=alpha, adjust=False, ignore_na=True).mean()
        mask_val = s_val.isna()
        imputed_val[mask_val.values, h] = smooth_val[mask_val].fillna(global_hourly_means[h]).values

    imputed_train = np.maximum(imputed_train, 0)
    imputed_val = np.maximum(imputed_val, 0)

    imputed_count_train = np.isnan(op_sales_masked_train).sum()
    imputed_count_val = np.isnan(op_sales_masked_val).sum()

    recovered_daily_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_daily_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    print(f"Imputed {imputed_count_train:,} hourly cells (train)")
    print(f"Imputed {imputed_count_val:,} hourly cells (val)")
    print(f"Alpha used: {alpha}")

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_kalman_like")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_kalman_like")

    return pd.concat([recovered_train_series, recovered_val_series]).sort_index()


def stl_real(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val, period=7):  # Laura - Seasonal and Trend decomposition using Loess
    imputed_train = op_sales_masked_train.copy()
    imputed_val = op_sales_masked_val.copy()
    n_hours = imputed_train.shape[1]

    global_hourly_means = np.nanmean(imputed_train, axis=0)  # globaler Fallback pro Stunde (nur train)

    def stl_column(col, fallback):
        s = pd.Series(col).astype(float)
        s_filled = s.interpolate(method="linear", limit_direction="both").fillna(fallback)

        mask = s.isna()
        try:
            stl = STL(s_filled, period=period, robust=True)
            result = stl.fit()
            estimate = result.trend + result.seasonal
            filled = s.copy()
            filled[mask] = estimate[mask].values
            return filled.values
        except Exception:
            filled = s.copy()
            filled[mask] = fallback
            return filled.values

    for h in range(n_hours):
        imputed_train[:, h] = stl_column(imputed_train[:, h], global_hourly_means[h])
        imputed_val[:, h] = stl_column(imputed_val[:, h], global_hourly_means[h])

    imputed_train = np.maximum(imputed_train, 0)
    imputed_val = np.maximum(imputed_val, 0)

    imputed_count_train = np.isnan(op_sales_masked_train).sum()
    imputed_count_val = np.isnan(op_sales_masked_val).sum()

    recovered_daily_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_daily_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    print(f"Imputed {imputed_count_train:,} hourly cells (train)")
    print(f"Imputed {imputed_count_val:,} hourly cells (val)")
    print(f"STL period used: {period}")

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_stl_real")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_stl_real")

    return pd.concat([recovered_train_series, recovered_val_series]).sort_index()


def stl_based(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val, period=7):
    """
    STL-nahe Imputation:
    - nutzt Rolling Median als Trend
    - nutzt Wochentagsmuster als Saison
    - füllt NaNs mit Trend + Saison
    """
    imputed_train = op_sales_masked_train.copy()
    imputed_val = op_sales_masked_val.copy()
    n_hours = imputed_train.shape[1]

    weekdays_train = pd.to_datetime(train["dt"]).dt.weekday.values
    weekdays_val = pd.to_datetime(val["dt"]).dt.weekday.values

    global_hourly_means = np.nanmean(imputed_train, axis=0)  # globaler Fallback pro Stunde (nur train)

    def trend_seasonal_column(col, weekdays, fallback):
        s = pd.Series(col)
        trend = s.rolling(window=period, min_periods=1, center=True).median()
        detrended = s - trend

        seasonal = np.zeros(len(s))
        for wd in range(7):
            wd_mask = weekdays == wd
            seasonal_value = np.nanmean(detrended[wd_mask])
            seasonal[wd_mask] = 0 if np.isnan(seasonal_value) else seasonal_value

        estimate = (trend + seasonal).fillna(fallback)
        mask = s.isna()
        filled = s.copy()
        filled[mask] = estimate[mask]
        return filled.values

    for h in range(n_hours):
        imputed_train[:, h] = trend_seasonal_column(imputed_train[:, h], weekdays_train, global_hourly_means[h])
        imputed_val[:, h] = trend_seasonal_column(imputed_val[:, h], weekdays_val, global_hourly_means[h])

    imputed_train = np.maximum(imputed_train, 0)
    imputed_val = np.maximum(imputed_val, 0)

    imputed_count_train = np.isnan(op_sales_masked_train).sum()
    imputed_count_val = np.isnan(op_sales_masked_val).sum()

    recovered_daily_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_daily_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    print(f"Imputed {imputed_count_train:,} hourly cells (train)")
    print(f"Imputed {imputed_count_val:,} hourly cells (val)")
    print(f"Period used: {period}")

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_stl_based")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_stl_based")

    return pd.concat([recovered_train_series, recovered_val_series]).sort_index()

# ML-basierte Recovery-Methoden: - Laura

def knn(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val, n_neighbors=5):  # KNN-Imputation - Laura
    imputed_train = op_sales_masked_train.copy()
    imputed_val = op_sales_masked_val.copy()

    imputed_count_train = np.isnan(imputed_train).sum()
    imputed_count_val = np.isnan(imputed_val).sum()

    imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    imputed_train = imputer.fit_transform(imputed_train)  # fit neighbor pool on train
    imputed_val = imputer.transform(imputed_val)           # val imputed using train's fitted neighbors

    imputed_train = np.maximum(imputed_train, 0)
    imputed_val = np.maximum(imputed_val, 0)

    recovered_daily_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_daily_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    print(f"Imputed {imputed_count_train:,} hourly cells (train)")
    print(f"Imputed {imputed_count_val:,} hourly cells (val)")
    print(f"KNN neighbors used: {n_neighbors}")

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_knn")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_knn")

    return pd.concat([recovered_train_series, recovered_val_series]).sort_index()


def _build_tree_features(df, imputed, rows, hours):
    """Shared feature builder for random_forest / lightgbm / xgboost."""
    dt_weekday = pd.to_datetime(df["dt"]).dt.weekday.values
    return pd.DataFrame({
        "hour": hours,
        "series_id": df["series_id"].values[rows],
        "day_idx": df["day_idx"].values[rows],
        "weekday": dt_weekday[rows],
        "discount": df["discount"].values[rows],
        "holiday_flag": df["holiday_flag"].values[rows],
        "activity_flag": df["activity_flag"].values[rows],
        "avg_temperature": df["avg_temperature"].fillna(0).values[rows],
        "avg_humidity": df["avg_humidity"].fillna(0).values[rows],
        "avg_wind_level": df["avg_wind_level"].fillna(0).values[rows],
        "precpt": df["precpt"].fillna(0).values[rows],
    })

def _fit_and_impute_tree_model(model, train, val, op_sales_masked_train, op_sales_masked_val,
                                max_train_rows, batch_size, random_state):
    """Shared train/predict/impute logic for random_forest, lightgbm, xgboost."""
    imputed_train = op_sales_masked_train.copy()
    imputed_val = op_sales_masked_val.copy()

    imputed_count_train = np.isnan(imputed_train).sum()
    imputed_count_val = np.isnan(imputed_val).sum()

    # ---- training data: observed hours in TRAIN only ----
    rows_obs, hours_obs = np.where(~np.isnan(imputed_train))
    X_obs = _build_tree_features(train, imputed_train, rows_obs, hours_obs)
    y_obs = imputed_train[rows_obs, hours_obs]

    print(f"Visible training rows: {len(X_obs):,}")

    if len(X_obs) > max_train_rows:
        sample_idx = np.random.default_rng(random_state).choice(len(X_obs), size=max_train_rows, replace=False)
        X_train_fit, y_train_fit = X_obs.iloc[sample_idx], y_obs[sample_idx]
    else:
        X_train_fit, y_train_fit = X_obs, y_obs

    print(f"Training rows used: {len(X_train_fit):,}")
    print("Training model...")
    start_fit = time.time()
    model.fit(X_train_fit, y_train_fit)
    print(f"Training finished in {time.time() - start_fit:.2f} seconds")

    # ---- predict missing cells for a given split, batched ----
    def predict_missing(df, imputed):
        rows_miss, hours_miss = np.where(np.isnan(imputed))
        print(f"Predicting missing rows: {len(rows_miss):,}")
        start_pred = time.time()
        for start in range(0, len(rows_miss), batch_size):
            end = min(start + batch_size, len(rows_miss))
            print(f"Predicting batch {start:,} to {end:,}")
            batch_rows, batch_hours = rows_miss[start:end], hours_miss[start:end]
            X_missing = _build_tree_features(df, imputed, batch_rows, batch_hours)
            preds = model.predict(X_missing)
            imputed[batch_rows, batch_hours] = preds
        print(f"Prediction finished in {time.time() - start_pred:.2f} seconds")
        return imputed

    imputed_train = predict_missing(train, imputed_train)
    imputed_val = predict_missing(val, imputed_val)

    imputed_train = np.maximum(imputed_train, 0)
    imputed_val = np.maximum(imputed_val, 0)

    print(f"Imputed {imputed_count_train:,} hourly cells (train)")
    print(f"Imputed {imputed_count_val:,} hourly cells (val)")

    return imputed_train, imputed_val

def random_forest(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val, max_train_rows=500_000, batch_size=500_000, random_state=42):  # Laura
    print("\n=== Random Forest Recovery ===")
    model = RandomForestRegressor(n_estimators=50, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=random_state)

    imputed_train, imputed_val = _fit_and_impute_tree_model(
        model, train, val, op_sales_masked_train, op_sales_masked_val, max_train_rows, batch_size, random_state
    )

    recovered_daily_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_daily_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_random_forest")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_random_forest")

    return pd.concat([recovered_train_series, recovered_val_series]).sort_index()

def lightgbm(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val, max_train_rows=500_000, batch_size=500_000, random_state=42):  # Laura
    print("\n=== LightGBM Recovery ===")
    model = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=-1, num_leaves=64, min_child_samples=50,
        subsample=0.8, colsample_bytree=0.8, objective="regression", n_jobs=-1,
        random_state=random_state, verbosity=-1,
    )

    imputed_train, imputed_val = _fit_and_impute_tree_model(
        model, train, val, op_sales_masked_train, op_sales_masked_val, max_train_rows, batch_size, random_state
    )

    recovered_daily_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_daily_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_lightgbm")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_lightgbm")

    return pd.concat([recovered_train_series, recovered_val_series]).sort_index()

def xgboost(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val, max_train_rows=500_000, batch_size=500_000, random_state=42):  # Laura
    print("\n=== XGBoost Recovery ===")
    model = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=8, min_child_weight=10, subsample=0.8,
        colsample_bytree=0.8, objective="reg:squarederror", n_jobs=-1,
        random_state=random_state, tree_method="hist",
    )

    imputed_train, imputed_val = _fit_and_impute_tree_model(
        model, train, val, op_sales_masked_train, op_sales_masked_val, max_train_rows, batch_size, random_state
    )

    recovered_daily_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_daily_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_xgboost")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_xgboost")

    return pd.concat([recovered_train_series, recovered_val_series]).sort_index()

def lightgbm_v2(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val, max_train_rows=1_500_000, batch_size=500_000, random_state=42):
    print("\n=== LightGBM Recovery v2 ===")

    imputed_train = op_sales_masked_train.copy()
    imputed_val = op_sales_masked_val.copy()
    imputed_count_train = np.isnan(imputed_train).sum()
    imputed_count_val = np.isnan(imputed_val).sum()

    def build_base(df, op_sales_masked, outside_slice):
        dt = pd.to_datetime(df["dt"])
        visible_sum = np.nansum(op_sales_masked, axis=1)
        visible_count = np.sum(~np.isnan(op_sales_masked), axis=1)
        visible_mean = visible_sum / np.maximum(visible_count, 1)
        daily_raw = df["sale_amount"].values if "sale_amount" in df.columns else np.zeros(len(df))

        base = pd.DataFrame({
            "series_id": df["series_id"].values,
            "day_idx": df["day_idx"].values,
            "weekday": dt.dt.weekday.values,
            "month": dt.dt.month.values,
            "discount": df["discount"].values,
            "holiday_flag": df["holiday_flag"].values,
            "activity_flag": df["activity_flag"].values,
            "avg_temperature": df["avg_temperature"].fillna(0).values,
            "avg_humidity": df["avg_humidity"].fillna(0).values,
            "avg_wind_level": df["avg_wind_level"].fillna(0).values,
            "precpt": df["precpt"].fillna(0).values,
            "outside_slice": outside_slice,
            "visible_sum": visible_sum,
            "visible_count": visible_count,
            "visible_mean": visible_mean,
            "daily_raw": daily_raw,
        })
        for col in ["product_id", "store_id", "city_id", "management_group_id", "psd"]:
            if col in df.columns:
                base[col] = df[col].values

        base["weekday_sin"] = np.sin(2 * np.pi * base["weekday"] / 7)
        base["weekday_cos"] = np.cos(2 * np.pi * base["weekday"] / 7)
        base["month_sin"] = np.sin(2 * np.pi * base["month"] / 12)
        base["month_cos"] = np.cos(2 * np.pi * base["month"] / 12)
        return base

    base_train = build_base(train, op_sales_masked_train, outside_slice_train)
    base_val = build_base(val, op_sales_masked_val, outside_slice_val)

    def build_X(base, rows, hours):
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

    # ---- training data: observed hours in TRAIN only ----
    rows_obs, hours_obs = np.where(~np.isnan(imputed_train))
    y_obs = imputed_train[rows_obs, hours_obs]
    print(f"Visible training rows: {len(rows_obs):,}")

    if len(rows_obs) > max_train_rows:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(len(rows_obs), size=max_train_rows, replace=False)
        rows_train, hours_train, y_train = rows_obs[sample_idx], hours_obs[sample_idx], y_obs[sample_idx]
    else:
        rows_train, hours_train, y_train = rows_obs, hours_obs, y_obs

    X_train = build_X(base_train, rows_train, hours_train)
    print(f"Training rows used: {len(X_train):,}")
    sample_weight = np.sqrt(y_train + 1)

    model = lgb.LGBMRegressor(
        objective="regression_l1", boosting_type="gbdt",
        n_estimators=700, learning_rate=0.035,
        num_leaves=127, max_depth=12, min_child_samples=40,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
        reg_alpha=0.5, reg_lambda=1.5,
        n_jobs=-1, random_state=random_state, verbosity=-1,
    )

    print("Training LightGBM v2...")
    start_fit = time.time()
    model.fit(X_train, y_train, sample_weight=sample_weight)
    print(f"Training finished in {time.time() - start_fit:.2f} seconds")

    def predict_missing(base, imputed):
        rows_miss, hours_miss = np.where(np.isnan(imputed))
        print(f"Predicting missing rows: {len(rows_miss):,}")
        start_pred = time.time()
        for start in range(0, len(rows_miss), batch_size):
            end = min(start + batch_size, len(rows_miss))
            print(f"Predicting batch {start:,} to {end:,}")
            batch_rows, batch_hours = rows_miss[start:end], hours_miss[start:end]
            X_missing = build_X(base, batch_rows, batch_hours)
            preds = np.clip(model.predict(X_missing), 0, None)
            imputed[batch_rows, batch_hours] = preds
        print(f"Prediction finished in {time.time() - start_pred:.2f} seconds")
        return imputed

    imputed_train = predict_missing(base_train, imputed_train)
    imputed_val = predict_missing(base_val, imputed_val)

    imputed_train = np.maximum(imputed_train, 0)
    imputed_val = np.maximum(imputed_val, 0)

    recovered_daily_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_daily_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    print(f"Imputed {imputed_count_train:,} hourly cells (train)")
    print(f"Imputed {imputed_count_val:,} hourly cells (val)")

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_lightgbm_v2")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_lightgbm_v2")

    return pd.concat([recovered_train_series, recovered_val_series]).sort_index()

def iterative(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val, max_iter=5, random_state=42):  # Laura
    print("\n=== Iterative Imputation Recovery ===")

    imputed_train = op_sales_masked_train.copy()
    imputed_val = op_sales_masked_val.copy()
    imputed_count_train = np.isnan(imputed_train).sum()
    imputed_count_val = np.isnan(imputed_val).sum()

    print(f"Max iterations: {max_iter}")

    estimator = ExtraTreesRegressor(n_estimators=30, max_depth=10, min_samples_leaf=20, n_jobs=-1, random_state=random_state)
    imputer = IterativeImputer(
        estimator=estimator, max_iter=max_iter, initial_strategy="mean",
        imputation_order="ascending", random_state=random_state, skip_complete=True, verbose=1,
    )

    print("Starting iterative imputation (fit on train)...")
    start_impute = time.time()
    imputed_train = imputer.fit_transform(imputed_train)  # fits the per-column estimators on train
    print(f"Train iterative imputation finished in {time.time() - start_impute:.2f} seconds")

    print("Applying fitted imputer to val...")
    start_val = time.time()
    imputed_val = imputer.transform(imputed_val)  # reuses train-fitted estimators, no refitting on val
    print(f"Val iterative imputation finished in {time.time() - start_val:.2f} seconds")

    imputed_train = np.maximum(imputed_train, 0)
    imputed_val = np.maximum(imputed_val, 0)

    recovered_daily_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_daily_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    print(f"Imputed {imputed_count_train:,} hourly cells (train)")
    print(f"Imputed {imputed_count_val:,} hourly cells (val)")

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_iterative")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_iterative")

    return pd.concat([recovered_train_series, recovered_val_series]).sort_index()

def iterative_improved(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val, max_iter=5, random_state=42):  # Laura
    print("\n=== Improved Iterative Imputation Recovery ===")

    imputed_train = op_sales_masked_train.copy()
    imputed_val = op_sales_masked_val.copy()
    imputed_count_train = np.isnan(imputed_train).sum()
    imputed_count_val = np.isnan(imputed_val).sum()

    estimator = ExtraTreesRegressor(n_estimators=50, max_depth=12, min_samples_leaf=10, n_jobs=-1, random_state=random_state)
    imputer = IterativeImputer(
        estimator=estimator, max_iter=max_iter, initial_strategy="mean",
        imputation_order="roman", random_state=random_state, skip_complete=True, verbose=1,
    )

    print("Starting iterative imputation (fit on train)...")
    imputed_train_new = imputer.fit_transform(imputed_train)
    missing_mask_train = np.isnan(op_sales_masked_train)
    imputed_train[missing_mask_train] = imputed_train_new[missing_mask_train]

    print("Applying fitted imputer to val...")
    imputed_val_new = imputer.transform(imputed_val)
    missing_mask_val = np.isnan(op_sales_masked_val)
    imputed_val[missing_mask_val] = imputed_val_new[missing_mask_val]

    imputed_train = np.maximum(imputed_train, 0)
    imputed_val = np.maximum(imputed_val, 0)

    recovered_daily_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_daily_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    print(f"Imputed {imputed_count_train:,} hourly cells (train)")
    print(f"Imputed {imputed_count_val:,} hourly cells (val)")

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_iterative_improved")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_iterative_improved")

    return pd.concat([recovered_train_series, recovered_val_series]).sort_index()

# Spezifische Demandrevovery Modelle: - Nils

def tobit(train): # TODO
    # ---------- FEATURES ----------
    hours_matrix = np.vstack(train["hours_sale"].values)

    # Summarise the 24-hour vector into 4 interpretable features
    # instead of 24 raw columns — reduces parameters from 32 to 12,
    # much better identified on typical product-level sample sizes
    hours_stock = np.vstack(train["hours_stock_status"].values)  # (n, 24)
    peak_hours   = slice(9, 21)   # 09:00–20:00, main selling window

    hours_features = pd.DataFrame({
        "peak_sales":      hours_matrix[:, peak_hours].sum(axis=1),
        "offpeak_sales":   hours_matrix[:, :9].sum(axis=1) + hours_matrix[:, 21:].sum(axis=1),
        "peak_stockout_h": hours_stock[:, peak_hours].sum(axis=1),   # key censoring severity
        "avail_frac":      1 - train["stock_hour6_22_cnt"].values / 16,  # fraction of 6-22 available
    }, index=train.index)

    base_df = pd.DataFrame({
        "weekday":     pd.to_datetime(train["dt"]).dt.dayofweek,
        "temperature": train["avg_temperature"],
        "humidity":    train["avg_humidity"],
        "wind":        train["avg_wind_level"],
        "precpt":      train["precpt"],        # was missing — strong demand driver
        "holiday":     train["holiday_flag"],
        "activity":    train["activity_flag"],
        "discount":    train["discount"],
        "const":       1.0,
    }, index=train.index).fillna(0)

    X           = pd.concat([base_df, hours_features], axis=1).values.astype(np.float64)
    y           = train["sale_amount"].values.astype(np.float64).ravel()
    is_censored = train["is_censored"].values.astype(bool).ravel()

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
    recovered_daily = np.where(is_censored, np.maximum(e_y_star, 0), y)

    print(f"Converged: {result.success} | {result.message}")
    print(f"sigma_hat: {sigma_hat:.4f}")
    return recovered_daily

def tobit_improved(train):  # Tobit / censored regression for stockout recovery - Laura === Running recovery method: tobit_improved at 2026-06-12 19:46:54.791141 ===
# Converged: False | STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT
# sigma_hat: 0.1123
# Mean raw sale_amount:  0.9986
# Mean recovered sales:  0.5498
# Gespeichert: [0.  0.  5.3 ... 4.2 2.2 2.1]
# Verarbeitungszeit:  1:10:42.476935

    print("\n=== Tobit Recovery Model ===")

    # ---------- FEATURES ----------
    hours_matrix = np.vstack(train["hours_sale"].values)
    hours_stock = np.vstack(train["hours_stock_status"].values)

    peak_hours = slice(9, 21)

    hours_features = pd.DataFrame({
        "peak_sales": hours_matrix[:, peak_hours].sum(axis=1),
        "offpeak_sales": hours_matrix[:, :9].sum(axis=1) + hours_matrix[:, 21:].sum(axis=1),
        "peak_stockout_h": hours_stock[:, peak_hours].sum(axis=1),
        "avail_frac": 1 - train["stock_hour6_22_cnt"].values / 16,
    }, index=train.index)

    base_df = pd.DataFrame({
        "weekday": pd.to_datetime(train["dt"]).dt.dayofweek,
        "temperature": train["avg_temperature"],
        "humidity": train["avg_humidity"],
        "wind": train["avg_wind_level"],
        "precpt": train["precpt"],
        "holiday": train["holiday_flag"],
        "activity": train["activity_flag"],
        "discount": train["discount"],
    }, index=train.index).fillna(0)

    X_df = pd.concat([base_df, hours_features], axis=1).fillna(0)

    y = train["sale_amount"].values.astype(np.float64).ravel()
    is_censored = train["is_censored"].values.astype(bool).ravel()

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
    result = minimize(neg_log_likelihood, params_init, method="L-BFGS-B",
        options={"maxiter": 3000, "ftol": 1e-7, "maxfun": 50000},)

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

    recovered_daily = np.where(is_censored, np.maximum(expected_if_censored, y), y)

    print(f"Converged: {result.success} | {result.message}")
    print(f"sigma_hat: {sigma_hat:.4f}")

    if not result.success:
        print("Warning: Tobit optimization did not fully converge.")

    return recovered_daily

def bayesian_old(train):  # Bayesisches Regressionsmodell mit Metropolis-Hastings MCMC # 7 min aber schlechter als raw_data
    # ---------- FEATURES (identisch zu Tobit) ----------
    hours_matrix = np.vstack(train["hours_sale"].values).astype(np.float32)
    hours_df = pd.DataFrame(hours_matrix, columns=[f"hour_{h}" for h in range(24)],
                            index=train.index)

    base_df = pd.DataFrame({
        "weekday":     pd.to_datetime(train["dt"]).dt.dayofweek.astype(np.float32),
        "temperature": train["avg_temperature"],
        "humidity":    train["avg_humidity"],
        "wind":        train["avg_wind_level"],
        "holiday":     train["holiday_flag"],
        "activity":    train["activity_flag"],
        "discount":    train["discount"],
        "const":       1.0,
    }, index=train.index).fillna(0)

    X           = pd.concat([base_df, hours_df], axis=1).values.astype(np.float32)
    y           = train["sale_amount"].values.astype(np.float32).ravel()
    is_censored = train["is_censored"].values.astype(bool).ravel()

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
    recovered_daily = np.where(is_censored, np.maximum(e_y_star, 0), y)

    print(f"Acceptance rate: {acceptance_rate:.3f} (ideal: 0.2–0.5)")
    print(f"sigma_hat: {sigma_hat:.4f}")

    return recovered_daily
    
def bayesian(train):  # Bayesisches Modell mit NUTS
    # ---------- FEATURES (identisch zu Tobit) ----------
    hours_matrix = np.vstack(train["hours_sale"].values).astype(np.float32)
    hours_df = pd.DataFrame(hours_matrix, columns=[f"hour_{h}" for h in range(24)],
                            index=train.index)

    base_df = pd.DataFrame({
        "weekday":     pd.to_datetime(train["dt"]).dt.dayofweek.astype(np.float32),
        "temperature": train["avg_temperature"],
        "humidity":    train["avg_humidity"],
        "wind":        train["avg_wind_level"],
        "holiday":     train["holiday_flag"],
        "activity":    train["activity_flag"],
        "discount":    train["discount"],
        "const":       1.0,
    }, index=train.index).fillna(0)

    X           = pd.concat([base_df, hours_df], axis=1).values.astype(np.float64)
    y           = train["sale_amount"].values.astype(np.float64).ravel()
    is_censored = train["is_censored"].values.astype(bool).ravel()

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
    recovered_daily = np.where(is_censored, np.maximum(e_y_star, 0), y)

    print(f"sigma_hat: {sigma_hat:.4f}")

    return recovered_daily

# Deep Learning basierte Recovery-Methoden: - Nils 

def autoencoder(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val,
                 latent_dim=8, epochs=20, lr=1e-3, batch_size=256, device=None):

    # Architecture: input = [16 sales + 16 mask flags + 6 covariates] = 38-dim
    # Encoder: 38 → 64 → 32 → latent_dim
    # Decoder: latent_dim → 32 → 64 → 16  (reconstructs all hours)
    # Loss: MSE on observed hours only, trained on TRAIN only

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    H = op_sales_masked_train.shape[1]

    # ── Covariate normalisation stats fit on TRAIN only ────────────────────────
    def fit_norm(x):
        return x.mean(), x.std() + 1e-8

    def apply_norm(x, mu, sigma):
        return (x - mu) / sigma

    mu_d, sd_d = fit_norm(train["discount"].values)
    mu_t, sd_t = fit_norm(train["avg_temperature"].values)
    mu_h, sd_h = fit_norm(train["avg_humidity"].values)
    mu_p, sd_p = fit_norm(train["precpt"].values)

    def build_covariates(df, mu_d, sd_d, mu_t, sd_t, mu_h, sd_h, mu_p, sd_p):
        return np.column_stack([
            apply_norm(df["discount"].values, mu_d, sd_d),
            df["holiday_flag"].values,
            df["activity_flag"].values,
            apply_norm(df["avg_temperature"].values, mu_t, sd_t),
            apply_norm(df["avg_humidity"].values, mu_h, sd_h),
            apply_norm(df["precpt"].values, mu_p, sd_p),
        ]).astype(np.float32)

    cov_train = build_covariates(train, mu_d, sd_d, mu_t, sd_t, mu_h, sd_h, mu_p, sd_p)  # (N_train, 6)
    cov_val = build_covariates(val, mu_d, sd_d, mu_t, sd_t, mu_h, sd_h, mu_p, sd_p)      # (N_val, 6)

    # ── Sales normalisation stats fit on TRAIN's observed hours only ──────────
    observed = op_sales_masked_train[~np.isnan(op_sales_masked_train)]
    sale_mean, sale_std = observed.mean(), observed.std() + 1e-8

    sales_norm_train = (op_sales_masked_train - sale_mean) / sale_std
    obs_mask_train = (~np.isnan(sales_norm_train)).astype(np.float32)
    sales_input_train = np.nan_to_num(sales_norm_train, nan=0.0).astype(np.float32)

    X_train = np.concatenate([sales_input_train, obs_mask_train, cov_train], axis=1)  # (N_train, 38)
    tgt_train = sales_norm_train.copy().astype(np.float32)

    T_X = torch.tensor(X_train)
    T_tgt = torch.tensor(tgt_train)
    T_obs = torch.tensor(obs_mask_train, dtype=torch.bool)

    loader = DataLoader(TensorDataset(T_X, T_tgt, T_obs), batch_size=batch_size, shuffle=True)

    # ── Model ───────────────────────────────────────────────────────────────
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
                nn.Linear(32, 64), nn.GELU(),
                nn.Linear(64, H),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = DemandAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Train (train only) ─────────────────────────────────────────────────────
    print(f"Training autoencoder on {device}  |  params: {sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_n = 0.0, 0
        for x, tgt_b, obs_b in loader:
            x, tgt_b, obs_b = x.to(device), tgt_b.to(device), obs_b.to(device)
            pred = model(x)
            loss = nn.functional.huber_loss(pred[obs_b], tgt_b[obs_b], delta=1.0)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * obs_b.sum().item()
            total_n += obs_b.sum().item()
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{epochs}  loss={total_loss/max(total_n,1):.5f}")

    # ── Inference helper, used for both train and val ──────────────────────
    def infer(op_sales_masked, cov):
        sales_norm = (op_sales_masked - sale_mean) / sale_std
        obs_mask = (~np.isnan(sales_norm)).astype(np.float32)
        sales_input = np.nan_to_num(sales_norm, nan=0.0).astype(np.float32)

        X = np.concatenate([sales_input, obs_mask, cov], axis=1)
        T_X_ = torch.tensor(X)

        model.eval()
        preds = []
        with torch.no_grad():
            for (x,) in DataLoader(TensorDataset(T_X_), batch_size=batch_size):
                preds.append(model(x.to(device)).cpu().numpy())
        preds_denorm = (np.concatenate(preds) * sale_std + sale_mean).clip(0)

        imputed = op_sales_masked.copy()
        nan_mask = np.isnan(imputed)
        imputed[nan_mask] = preds_denorm[nan_mask]
        return imputed, nan_mask.sum()

    imputed_train, imputed_count_train = infer(op_sales_masked_train, cov_train)
    imputed_val, imputed_count_val = infer(op_sales_masked_val, cov_val)

    recovered_daily_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_daily_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    print(f"Imputed {imputed_count_train:,} hourly cells (train)")
    print(f"Imputed {imputed_count_val:,} hourly cells (val)")

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_autoencoder")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_autoencoder")

    return pd.concat([recovered_train_series, recovered_val_series]).sort_index()
#def transformer(train): # SAITS, BRITS, GRIN, CSDI

def transformer(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val, d_model=32, n_heads=4, n_layers=2, d_ff=64, epochs=20, lr=3e-4,
                 batch_size=256, device=None):
    """
    Imputes censored (NaN) hourly cells in op_sales_masked_train/val using an
    encoder-only Transformer trained ONLY on train's observed (non-NaN) hours,
    then applied for inference on both train and val.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    H = op_sales_masked_train.shape[1]

    def build_covariates(df, N):
        hour_idx = np.tile(np.arange(H), (N, 1)).astype(np.float32) / (H - 1)
        discount = np.repeat(df["discount"].values[:, None], H, axis=1).astype(np.float32)
        holiday = np.repeat(df["holiday_flag"].values[:, None], H, axis=1).astype(np.float32)
        activity = np.repeat(df["activity_flag"].values[:, None], H, axis=1).astype(np.float32)
        temperature = np.repeat(df["avg_temperature"].values[:, None], H, axis=1).astype(np.float32)
        humidity = np.repeat(df["avg_humidity"].values[:, None], H, axis=1).astype(np.float32)
        precpt = np.repeat(df["precpt"].values[:, None], H, axis=1).astype(np.float32)
        return hour_idx, discount, holiday, activity, temperature, humidity, precpt

    N_train = len(train)
    N_val = len(val)

    hour_idx_tr, discount_tr, holiday_tr, activity_tr, temp_tr, hum_tr, precpt_tr = build_covariates(train, N_train)
    hour_idx_v, discount_v, holiday_v, activity_v, temp_v, hum_v, precpt_v = build_covariates(val, N_val)

    # Normalisation stats fit on TRAIN only, applied to both
    def fit_norm(x):
        mu, sigma = x.mean(), x.std() + 1e-8
        return mu, sigma

    def apply_norm(x, mu, sigma):
        return (x - mu) / sigma

    mu_d, sd_d = fit_norm(discount_tr)
    mu_t, sd_t = fit_norm(temp_tr)
    mu_h, sd_h = fit_norm(hum_tr)
    mu_p, sd_p = fit_norm(precpt_tr)

    covariates_train = np.stack([
        hour_idx_tr, apply_norm(discount_tr, mu_d, sd_d), holiday_tr, activity_tr,
        apply_norm(temp_tr, mu_t, sd_t), apply_norm(hum_tr, mu_h, sd_h), apply_norm(precpt_tr, mu_p, sd_p),
    ], axis=-1)  # (N_train, H, C)

    covariates_val = np.stack([
        hour_idx_v, apply_norm(discount_v, mu_d, sd_d), holiday_v, activity_v,
        apply_norm(temp_v, mu_t, sd_t), apply_norm(hum_v, mu_h, sd_h), apply_norm(precpt_v, mu_p, sd_p),
    ], axis=-1)  # (N_val, H, C)

    C = covariates_train.shape[-1]

    # Sales normalisation stats fit on TRAIN's observed hours only
    observed_vals = op_sales_masked_train[~np.isnan(op_sales_masked_train)]
    sale_mean = observed_vals.mean()
    sale_std = observed_vals.std() + 1e-8

    sales_norm_train = (op_sales_masked_train - sale_mean) / sale_std
    obs_mask_train = ~np.isnan(sales_norm_train)
    sales_input_train = np.nan_to_num(sales_norm_train, nan=0.0).astype(np.float32)

    T_sales = torch.tensor(sales_input_train, dtype=torch.float32)
    T_cov = torch.tensor(covariates_train, dtype=torch.float32)
    T_obs = torch.tensor(obs_mask_train, dtype=torch.bool)
    T_tgt = torch.tensor(sales_norm_train.copy().astype(np.float32))

    dataset = TensorDataset(T_sales, T_cov, T_obs, T_tgt)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    class HourlyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(1 + C, d_model)

            pe = torch.zeros(H, d_model)
            pos = torch.arange(H).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=0.1, batch_first=True, activation="gelu"
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Linear(d_model, 1)

        def forward(self, sale, cov):
            x = torch.cat([sale.unsqueeze(-1), cov], dim=-1)
            x = self.input_proj(x) + self.pe
            x = self.encoder(x)
            return self.head(x).squeeze(-1)

    model = HourlyTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Training transformer on {device}  |  params: {sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_tokens = 0.0, 0
        for sale, cov, obs, tgt in loader:
            sale, cov, obs, tgt = sale.to(device), cov.to(device), obs.to(device), tgt.to(device)
            pred = model(sale, cov)

            if obs.sum() == 0:
                continue
            loss = nn.functional.huber_loss(pred[obs], tgt[obs], delta=1.0)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * obs.sum().item()
            epoch_tokens += obs.sum().item()

        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{epochs}  loss={epoch_loss/max(epoch_tokens,1):.5f}")

    # Inference helper, used for both train and val
    def infer(sales_masked, covariates):
        sales_norm = (sales_masked - sale_mean) / sale_std
        sales_input = np.nan_to_num(sales_norm, nan=0.0).astype(np.float32)

        T_s = torch.tensor(sales_input, dtype=torch.float32)
        T_c = torch.tensor(covariates, dtype=torch.float32)

        model.eval()
        all_preds = []
        inf_loader = DataLoader(TensorDataset(T_s, T_c), batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for sale, cov in inf_loader:
                pred = model(sale.to(device), cov.to(device))
                all_preds.append(pred.cpu().numpy())

        preds_norm = np.concatenate(all_preds, axis=0)
        preds_denorm = (preds_norm * sale_std + sale_mean).clip(0)

        imputed = sales_masked.copy()
        nan_mask = np.isnan(imputed)
        imputed[nan_mask] = preds_denorm[nan_mask]
        return imputed, nan_mask.sum()

    imputed_train, imputed_count_train = infer(op_sales_masked_train, covariates_train)
    imputed_val, imputed_count_val = infer(op_sales_masked_val, covariates_val)

    recovered_daily_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_daily_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    print(f"Imputed {imputed_count_train:,} hourly cells (train)")
    print(f"Imputed {imputed_count_val:,} hourly cells (val)")

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_transformer")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_transformer")

    return pd.concat([recovered_train_series, recovered_val_series]).sort_index()

def diffusion(train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val, noise_scale=0.1, n_samples=5, random_state=42):
    # Diffusion-like Recovery - Laura
    print("\n=== Diffusion-like Recovery ===")

    rng = np.random.default_rng(random_state)

    imputed_train = op_sales_masked_train.copy()
    imputed_val = op_sales_masked_val.copy()

    imputed_count_train = np.isnan(imputed_train).sum()
    imputed_count_val = np.isnan(imputed_val).sum()

    print(f"Matrix shape (train): {imputed_train.shape}")
    print(f"Matrix shape (val): {imputed_val.shape}")
    print(f"Missing values (train): {imputed_count_train:,}")
    print(f"Missing values (val): {imputed_count_val:,}")
    print(f"Number of samples: {n_samples}")
    print(f"Noise scale: {noise_scale}")

    # Hour mean/std fit on TRAIN only
    hour_mean = np.nanmean(imputed_train, axis=0)
    hour_std = np.nanstd(imputed_train, axis=0)
    hour_std = np.where(np.isnan(hour_std) | (hour_std == 0), 1e-6, hour_std)

    def sample_fill(imputed, nan_mask):
        sampled_values = [
            np.maximum(hour_mean + rng.normal(loc=0, scale=noise_scale, size=imputed.shape) * hour_std, 0)
            for _ in range(n_samples)
        ]
        generated = np.mean(sampled_values, axis=0)
        imputed = imputed.copy()
        imputed[nan_mask] = generated[nan_mask]
        return imputed

    nan_mask_train = np.isnan(imputed_train)
    nan_mask_val = np.isnan(imputed_val)

    imputed_train = sample_fill(imputed_train, nan_mask_train)
    imputed_val = sample_fill(imputed_val, nan_mask_val)

    recovered_daily_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_daily_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    print(f"Imputed {imputed_count_train:,} hourly cells (train)")
    print(f"Imputed {imputed_count_val:,} hourly cells (val)")

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_diffusion")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_diffusion")

    return pd.concat([recovered_train_series, recovered_val_series]).sort_index()

# DLinear (Nils)
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

    def __init__(self, hourly_sales, mask_prob=0.30, keep_min_hours=8):

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


def dlinear_train(op_sales_masked_train, epochs=50, batch_size=256, lr=1e-3,
                   mask_prob=0.30, device=None, model_path="dlinear_model.pt"):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Keep only complete (fully-observed) days from train for supervised training
    complete_days = ~np.isnan(op_sales_masked_train).any(axis=1)
    train_data = op_sales_masked_train[complete_days]

    print(f"Training days: {len(train_data):,}")

    dataset = RecoveryDataset(train_data, mask_prob=mask_prob)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    seq_len = op_sales_masked_train.shape[1]
    model = DLinear(seq_len=seq_len)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to '{model_path}'")

    return model


def dlinear(train, val, op_sales_masked_train, op_sales_masked_val,
            outside_slice_train, outside_slice_val, model_path="dlinear_model.pt"):
    """
    Recover hourly demand using a DLinear model, trained on train only.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seq_len = op_sales_masked_train.shape[1]
    model = DLinear(seq_len=seq_len)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model = dlinear_train(op_sales_masked_train, model_path=model_path)

    model.to(device)
    model.eval()

    def impute(op_sales_masked):
        imputed = op_sales_masked.copy()
        nan_mask = np.isnan(imputed)

        # DLinear cannot handle NaNs
        model_input = np.nan_to_num(imputed, nan=0.0)
        x = torch.tensor(model_input, dtype=torch.float32, device=device)

        with torch.no_grad():
            prediction = model(x).cpu().numpy()

        imputed[nan_mask] = prediction[nan_mask]
        imputed = np.maximum(imputed, 0)
        return imputed, nan_mask.sum()

    imputed_train, imputed_count_train = impute(op_sales_masked_train)
    imputed_val, imputed_count_val = impute(op_sales_masked_val)

    recovered_daily_train = outside_slice_train + np.nansum(imputed_train, axis=1)
    recovered_daily_val = outside_slice_val + np.nansum(imputed_val, axis=1)

    print(f"Imputed {imputed_count_train:,} hourly cells (train)")
    print(f"Imputed {imputed_count_val:,} hourly cells (val)")
    print(f"recovered_daily_train_mean: {np.nanmean(recovered_daily_train):.4f}")
    print(f"recovered_daily_val_mean: {np.nanmean(recovered_daily_val):.4f}")

    recovered_train_series = pd.Series(recovered_daily_train, index=train.index, name="recovered_daily_sales_dlinear")
    recovered_val_series = pd.Series(recovered_daily_val, index=val.index, name="recovered_daily_sales_dlinear")

    return pd.concat([recovered_train_series, recovered_val_series]).sort_index()