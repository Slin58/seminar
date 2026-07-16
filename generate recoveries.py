import numpy as np
import pandas as pd
import utils
import recovery
from datasets import load_dataset

import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

from pathlib import Path

token_file = Path("hugging_face_token.txt")
if token_file.exists():
    huggingface_token = token_file.read_text(encoding="utf-8").strip()
    ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K", token=huggingface_token,)    
else:
    ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K")

# Data preparation
train_raw = ds["train"].to_pandas()
eval_raw = ds["eval"].to_pandas()

history = utils.prepare_panel(train_raw)
history = utils.flag_censoring(history)
history = utils.make_features(history)

train, val = utils.time_split(history, horizon=7)
# TODO Hier drunter sollte nur mit train (nicht history) gearbeitet werden, da sonst data leakage entsteht, da die recoveries auf dem gesamten history berechnet werden und somit auch die future values enthalten sind.

series_stockouts = train.groupby("series_id")["is_censored"].mean()
example_sid = series_stockouts[(series_stockouts > 0.3) & (series_stockouts < 0.7)].index[0]

train["datum"] = pd.to_datetime(train["dt"])
train["weekday"] = train["datum"].dt.day_name()

# calculate outside_slice
hourly_sales = np.stack(train["hours_sale"].values)          # (N, 24)
hourly_stock_ds = np.stack(train["hours_stock_status"].values)  # (N, 24)

op_sales = hourly_sales[:, 6:22].astype(np.float32)
op_stock_status = hourly_stock_ds[:, 6:22].astype(np.float32)

op_sales_masked = np.where(op_stock_status == 1, np.nan, op_sales) # hours_sale, but stockout sales are censored

total_cells = op_sales_masked.size
missing_cells = np.isnan(op_sales_masked).sum()
print(f"Operating window: {op_sales_masked.shape[1]} hours (h06-h21)")
print(f"Missing hourly cells: {missing_cells:,} / {total_cells:,} ({missing_cells/total_cells:.1%})")

#visible_sum = np.nansum(np.where(op_stock_status == 0, op_sales, 0), axis=1) # all sales where enough stock was available
#outside_slice = np.maximum(history["sale_amount"].values.astype(np.float32) - visible_sum, 0) # sales that are in sale_amount but not in hours_sale due to the time frame (6-21) TODO möglicher Fehler Doppelzählung
outside_slice = np.maximum(train["sale_amount"].values.astype(np.float32) - np.nansum(op_sales, axis=1), 0) # sales that are in sale_amount but not in hours_sale due to the time frame (6-21)
# TODO history oder train


RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ------------------------------------------------------------
# 1. Recovery-Methoden registrieren
# ------------------------------------------------------------

recovery_methods = {
    "random_sampling": {
        "func": recovery.random_sampling,
        "args": (train, op_sales_masked, outside_slice, rng),
        "target_col": "recovered_daily_sales_random_sampling",
    },
    "global_mean": {
        "func": recovery.global_mean,
        "args": (train, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_global_mean",
    },
    "per_series_mean": {
        "func": recovery.per_series_mean,
        "args": (train,),
        "target_col": "recovered_daily_sales_per_series_mean",
    },
    "hourly_mean": {
        "func": recovery.hourly_mean,
        "args": (train, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_hourly_mean",
    },
    "hour_per_series_mean": {
        "func": recovery.hour_per_series_mean,
        "args": (train, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_hour_per_series_mean",
    },
    "weekday_mean": {
        "func": recovery.weekday_mean,
        "args": (train, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_weekday_mean",
    },
    "weekday_daily_mean": {
        "func": recovery.weekday_daily_mean,
        "args": (train,),
        "target_col": "recovered_daily_sales_weekday_daily_mean",
    },
    "rolling_mean": {
        "func": recovery.rolling_mean,
        "args": (train, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_rolling_mean",
    },
    "exponential_moving_average": {
        "func": recovery.exponential_moving_average,
        "args": (train, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_exponential_moving_average",
    },
    "exponential_moving_average_series": {
        "func": recovery.exponential_moving_average_series,
        "args": (train, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_exponential_moving_average_series",
    },
    "interpolation_linear": {
        "func": recovery.interpolation_linear,
        "args": (train, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_interpolation_linear",
    },
    "interpolation_spline": {
        "func": recovery.interpolation_spline,
        "args": (train, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_interpolation_spline",
    },

    "interpolation_spline_series": {
        "func": recovery.interpolation_spline_series,
        "args": (train, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_interpolation_spline_series",
    },


    "interpolation_polynomial": {
        "func": recovery.interpolation_polynomial,
        "args": (train, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_interpolation_polynomial",
    },
    "kalman_smoothing": { # 1:45 h
        "func": recovery.kalman_smoothing,
        "args": (history, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_kalman_smoothing",
    },
    "kalman_like": {
        "func": recovery.kalman_like_smoothing,
        "args": (train, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_kalman_like",
    },
    "stl_real": { # 1:21 h
        "func": recovery.stl_real,
        "args": (history, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_stl_real",
    },
    "stl_based": {
        "func": recovery.stl_based,
        "args": (train, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_stl_based",
    },
    # "knn": { # nicht durchgelaufen
    #     "func": recovery.knn,
    #     "args": (history, op_sales_masked, outside_slice),
    #     "target_col": "recovered_daily_sales_knn",
    # },
    "random_forest": {
        "func": recovery.random_forest,
        "args": (train, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_random_forest",
    },

    "lightgbm": {
        "func": recovery.lightgbm,
        "args": (train, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_lightgbm",
    },
    "xgboost": {
        "func": recovery.xgboost,
        "args": (train, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_xgboost",
    },
    "iterative": { # 1:37 h
        "func": recovery.iterative,
        "args": (history, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_iterative",
    },

    "iterative_improved": {
        "func": recovery.iterative_improved,
        "args": (train, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_iterative_improved",
    },

    "transformer": { # 2:13h 
        "func": recovery.transformer,
        "args": (history, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_transformer",
    },
    "diffusion": {
        "func": recovery.diffusion,
        "args": (train, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_diffusion",
    },
    "tobit": { # 1:10 h
        "func": recovery.tobit,
        "args": (history,),
        "target_col": "recovered_daily_sales_tobit",
    },

    "tobit_improved": { # 1:10 h
        "func": recovery.tobit_improved,
        "args": (history,),
        "target_col": "recovered_daily_sales_tobit_improved",
    },

    # "bayesian": { # nicht fertig
    #     "func": recovery.bayesian,
    #     "args": (history,),
    #     "target_col": "recovered_daily_sales_bayesian",  
    # },
    "autoencoder": {
        "func": recovery.autoencoder,
        "args": (train, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_autoencoder",
    },
    "dlinear": {
        "func": recovery.dlinear,
        "args": (train, op_sales, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_dlinear",
    },
    "lightgbm_v2": {
        "func": recovery.lightgbm_v2,
        "args": (train, op_sales_masked, outside_slice),
        "target_col": "recovered_daily_sales_lightgbm_v2",
    }, 
    #=== LightGBM Recovery v2 Finished ===
    # Imputed 14,311,536 hourly cells
    # Mean raw sale_amount: 0.9986
    # Mean recovered sales: 1.1762
    # Total runtime: 803.53 seconds
    # Gespeichert: [2.3159428  0.55325204 5.3        ... 3.8        2.2        2.1       ]
    # Verarbeitungszeit:  0:13:23.918970
}
# TODO Kosten Nutzen?
# knn über 3h
# bayesian hat zu lange gedauert 
# tobit: 0:42 h -> 0.5437, aber Converged: False | STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT 
# Transformer
# Autoencoder
# XGBoost Recovery
# LightGBM Recovery
# Random Forest Recovery
# iterative
# interpolation spline series
# interpolation polynomial

# transformer: 2:13 h -> 1.1008
# kalman-smoothing: 1:38 h -> mean recovered sales: 1.1415
# stl real: 1:18 h -> mean recoevered sales: 1.0977
# autoencoder: 0:18 h -> 1.0779

# ------------------------------------------------------------
# 2. Alle Recovery-Methoden ausführen
# ------------------------------------------------------------


from datetime import datetime
import json

for recovery_name, method in recovery_methods.items():
    current_time = datetime.now()

    print(f"\n=== Running recovery method: {recovery_name} at {current_time} ===")
    recovered_daily = method["func"](*method["args"])
    train[method["target_col"]] = recovered_daily
    print(f"Mean raw sale_amount: {train['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {recovered_daily.mean():.4f}")


    arr = train[f"{method['target_col']}"].to_numpy()

    np.save(f"recovered_column/{method['target_col']}.npy", arr)
    print("Gespeichert:", arr)

    processing_time = datetime.now()-current_time

    print("Verarbeitungszeit: ", processing_time)

    with open("recovered_column/recovery_processing_time.json", "r") as f:
        content = f.read()
        time = json.loads(content) if content.strip() else {}

    time[method['target_col']] = processing_time.total_seconds()

    with open("recovered_column/recovery_processing_time.json", "w") as f:
        json.dump(time, f)

