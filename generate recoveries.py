import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
import importlib
import utils
import recovery
import os
os.system("pip install -q pandas pyarrow matplotlib seaborn datasets") # TODO
from datasets import load_dataset


ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K")

# Data preparation
train_raw = ds["train"].to_pandas()
eval_raw = ds["eval"].to_pandas()

history = utils.prepare_panel(train_raw)
history = utils.flag_censoring(history)
history = utils.make_features(history)

train, val = utils.time_split(history, horizon=7)


series_stockouts = history.groupby("series_id")["is_censored"].mean()
example_sid = series_stockouts[(series_stockouts > 0.3) & (series_stockouts < 0.7)].index[0]

history["datum"] = pd.to_datetime(history["dt"])
history["weekday"] = history["datum"].dt.day_name()

# calculate outside_slice
hourly_sales = np.stack(history["hours_sale"].values)          # (N, 24)
hourly_stock_ds = np.stack(history["hours_stock_status"].values)  # (N, 24)

op_sales = hourly_sales[:, 6:22].astype(np.float32)
op_stock_status = hourly_stock_ds[:, 6:22].astype(np.float32)

op_sales_masked = np.where(op_stock_status == 1, np.nan, op_sales) # hours_sale, but stockout sales are censored

total_cells = op_sales_masked.size
missing_cells = np.isnan(op_sales_masked).sum()

visible_sum = np.nansum(np.where(op_stock_status == 0, op_sales, 0), axis=1) # all sales where enough stock was available
#outside_slice = np.maximum(history["sale_amount"].values.astype(np.float32) - visible_sum, 0) # sales that are in sale_amount but not in hours_sale due to the time frame (6-21) TODO möglicher Fehler Doppelzählung
outside_slice = np.maximum(history["sale_amount"].values.astype(np.float32) - np.nansum(op_sales), 0) # sales that are in sale_amount but not in hours_sale due to the time frame (6-21)



# ------------------------------------------------------------
# 1. Recovery-Methoden registrieren
# ------------------------------------------------------------

recovery_methods = {
    # "random_sampling": {
    #     "func": recovery.random_sampling,
    #     "args": (history, op_sales_masked, outside_slice, rng),
    #     "target_col": "recovered_daily_sales_random_sampling",
    # },

    "global_mean": {
        "func": recovery.global_mean,
        "args": (history, op_stock_status, op_sales, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_global_mean",
    },

    # "per_series_mean": {
    #     "func": recovery.per_series_mean,
    #     "args": (history,),
    #     "target_col": "recovered_daily_sales_per_series_mean",
    # },

    # "seasonal_mean": {
    #     "func": recovery.seasonal_mean,
    #     "args": (history,),
    #     "target_col": "recovered_daily_sales_seasonal_mean",
    # },

    # "hour_per_seasonal_mean": {
    #     "func": recovery.hour_per_seasonal_mean,
    #     "args": (history, op_sales_masked, outside_slice),
    #     "target_col": "recovered_daily_sales_hour_per_seasonal_mean",
    # },

    # "hourly_mean": {
    #     "func": recovery.hourly_mean,
    #     "args": (history, op_sales_masked, outside_slice),
    #     "target_col": "recovered_daily_sales_hourly_mean",
    # },

    # "hour_per_series_mean": {
    #     "func": recovery.hour_per_series_mean,
    #     "args": (history, op_sales_masked, outside_slice),
    #     "target_col": "recovered_daily_sales_hour_per_series_mean",
    # },

    # "weekday_mean": {
    #     "func": recovery.weekday_mean,
    #     "args": (history, op_sales_masked, outside_slice),
    #     "target_col": "recovered_daily_sales_weekday_mean",
    # },

    # "rolling_mean": {
    #     "func": recovery.rolling_mean,
    #     "args": (history, op_sales_masked, outside_slice),
    #     "target_col": "recovered_daily_sales_rolling_mean",
    # },

    "ema": {
        "func": recovery.exponential_moving_average,
        "args": (history, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_exponential_moving_average",
    },

    # "ema_series": {
    #     "func": recovery.exponential_moving_average_series,
    #     "args": (history, op_sales_masked, outside_slice),
    #     "target_col": "recovered_daily_sales_exponential_moving_average_series",
    # },

    # "tobit_model": {
    #     "func": recovery.tobit_model,
    #     "args": (history,),
    #     "target_col": "recovered_daily_sales_tobit",
    #},

    # "bayesian_model": {
    #     "func": recovery.bayesian_model,
    #     "args": (history,),
    #     "target_col": "recovered_daily_sales_bayesian",  
    # },

    # "interpolation_linear": {
    #     "func": recovery.interpolation_linear,
    #     "args": (history, op_sales_masked, outside_slice),
    #     "target_col": "recovered_daily_sales_interpolation_linear",
    # },

    # "interpolation_spline": {
    #     "func": recovery.interpolation_spline,
    #     "args": (history, op_sales_masked, outside_slice),
    #     "target_col": "recovered_daily_sales_interpolation_spline",
    # },

    # "interpolation_polynomial": {
    #     "func": recovery.interpolation_polynomial,
    #     "args": (history, op_sales_masked, outside_slice),
    #     "target_col": "recovered_daily_sales_interpolation_polynomial",
    # },

    # "kalman_smoothing": {
    #     "func": recovery.kalman_smoothing,
    #     "args": (history, op_sales_masked, outside_slice),
    #     "target_col": "recovered_daily_sales_kalman_smoothing",
    # },

    # "kalman_like": {
    #     "func": recovery.kalman_like_smoothing,
    #     "args": (history, op_sales_masked, outside_slice),
    #     "target_col": "recovered_daily_sales_kalman_like",
    # },

    # "stl_real": {
    #     "func": recovery.stl_real,
    #     "args": (history, op_sales_masked, outside_slice),
    #     "target_col": "recovered_daily_sales_stl_real",
    # },

    # "stl_based": {
    #     "func": recovery.stl_based,
    #     "args": (history, op_sales_masked, outside_slice),
    #     "target_col": "recovered_daily_sales_stl_based",
    # },

    # "knn": {
    #     "func": recovery.knn,
    #     "args": (history, op_sales_masked, outside_slice),
    #     "target_col": "recovered_daily_sales_knn",
    # },

    # "autoencoder": {
    #     "func": recovery.autoencoder,
    #     "args": (history, op_sales_masked, outside_slice),
    #     "target_col": "recovered_daily_sales_autoencoder",
    # },

}

# ------------------------------------------------------------
# 2. Alle Recovery-Methoden ausführen
# ------------------------------------------------------------

for recovery_name, method in recovery_methods.items():
    print(f"\n=== Running recovery method: {recovery_name} ===")
    method["func"](*method["args"])
    arr = history[f"{method['target_col']}"].to_numpy()

    np.save(f"recovered_column/{method['target_col']}.npy", arr)
    print("Gespeichert:", arr)


