import numpy as np
import pandas as pd
import utils
import recovery_copy
from datasets import load_dataset
from pathlib import Path
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

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

train["datum"] = pd.to_datetime(train["dt"])
train["weekday"] = train["datum"].dt.day_name()
val["datum"] = pd.to_datetime(val["dt"])
val["weekday"] = val["datum"].dt.day_name()

# calculate outside_slice
hourly_sales_train = np.stack(train["hours_sale"].values)          # (N, 24)
hourly_stock_ds_train = np.stack(train["hours_stock_status"].values)  # (N, 24)

op_sales_train = hourly_sales_train[:, 6:22].astype(np.float32)
op_stock_status_train = hourly_stock_ds_train[:, 6:22].astype(np.float32)

op_sales_masked_train = np.where(op_stock_status_train == 1, np.nan, op_sales_train) # hours_sale, but stockout sales are censored

total_cells_train = op_sales_masked_train.size
missing_cells_train = np.isnan(op_sales_masked_train).sum()
print(f"Operating window: {op_sales_masked_train.shape[1]} hours (h06-h21)")
print(f"Missing hourly cells: {missing_cells_train:,} / {total_cells_train:,} ({missing_cells_train/total_cells_train:.1%})")

outside_slice_train = np.maximum(train["sale_amount"].values.astype(np.float32) - np.nansum(op_sales_train, axis=1), 0) # sales that are in sale_amount but not in hours_sale due to the time frame (6-21)

val["datum"] = pd.to_datetime(val["dt"])
val["weekday"] = val["datum"].dt.day_name()
# calculate outside_slice
hourly_sales_val = np.stack(val["hours_sale"].values)          # (N, 24)
hourly_stock_ds_val = np.stack(val["hours_stock_status"].values)  # (N, 24)

op_sales_val = hourly_sales_val[:, 6:22].astype(np.float32)
op_stock_status_val = hourly_stock_ds_val[:, 6:22].astype(np.float32)

op_sales_masked_val = np.where(op_stock_status_val == 1, np.nan, op_sales_val) # hours_sale, but stockout sales are censored

total_cells_val = op_sales_masked_val.size
missing_cells_val = np.isnan(op_sales_masked_val).sum()
print(f"Operating window: {op_sales_masked_val.shape[1]} hours (h06-h21)")
print(f"Missing hourly cells: {missing_cells_val:,} / {total_cells_val:,} ({missing_cells_val/total_cells_val:.1%})")

outside_slice_val = np.maximum(val["sale_amount"].values.astype(np.float32) - np.nansum(op_sales_val, axis=1), 0) # sales that are in sale_amount but not in hours_sale due to the time frame (6-21)


RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ------------------------------------------------------------
# 1. Recovery-Methoden registrieren
# ------------------------------------------------------------

recovery_methods = {
    # "random_sampling": {
    #     "func": recovery_copy.random_sampling,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val, rng),
    # },
    # "global_mean": {
    #     "func": recovery_copy.global_mean,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },
    # "hourly_mean": {
    #     "func": recovery_copy.hourly_mean,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },
    # "series_mean": {
    #     "func": recovery_copy.series_mean,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },
    # "series_daily_mean": {
    #     "func": recovery_copy.series_daily_mean,
    #     "args": (train, val),
    # },
    # "weekday_mean": {
    #     "func": recovery_copy.weekday_mean,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },
    # "weekday_daily_mean": {
    #     "func": recovery_copy.weekday_daily_mean,
    #     "args": (train, val,),
    # },
    # "rolling_mean": {
    #     "func": recovery_copy.rolling_mean,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },
    # "exponential_moving_average": {
    #     "func": recovery_copy.exponential_moving_average,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },
    # "exponential_moving_average_series": {
    #     "func": recovery_copy.exponential_moving_average_series,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },
    # "interpolation_linear": {
    #     "func": recovery_copy.interpolation_linear,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },
    # "interpolation_spline": {
    #     "func": recovery_copy.interpolation_spline,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },

    # "interpolation_spline_series": {
    #     "func": recovery_copy.interpolation_spline_series,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },


    # "interpolation_polynomial": {
    #     "func": recovery_copy.interpolation_polynomial,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },
    # "kalman_smoothing": { # 1:45 h
    #     "func": recovery_copy.kalman_smoothing,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },
    # "kalman_like": {
    #     "func": recovery_copy.kalman_like_smoothing,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },
    # "stl_real": { # 1:21 h
    #     "func": recovery_copy.stl_real,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },
    # "stl_based": {
    #     "func": recovery_copy.stl_based,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },
    # # "knn": { # nicht durchgelaufen
    # #     "func": recovery_copy.knn,
    # #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # # },
    # "random_forest": {
    #     "func": recovery_copy.random_forest,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },

    # "lightgbm": {
    #     "func": recovery_copy.lightgbm,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },
    # "xgboost": {
    #     "func": recovery_copy.xgboost,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },
    # "iterative": { # 1:37 h
    #     "func": recovery_copy.iterative,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },

    # "iterative_improved": { # nicht durchgelaufen (fast 5 h)
    #     "func": recovery_copy.iterative_improved,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },

    # "transformer": { # 2:13h 
    #     "func": recovery_copy.transformer,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },
    # "diffusion": {
    #     "func": recovery_copy.diffusion,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },
    # # "tobit": { # 1:10 h
    # #     "func": recovery_copy.tobit,
    # #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # # },

    # # "tobit_improved": { # 1:10 h
    # #     "func": recovery_copy.tobit_improved,
    # #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # # },

    # # "bayesian": { # nicht fertig
    # #     "func": recovery_copy.bayesian,
    # #     "args": (history,),
    # # },
    # "autoencoder": {
    #     "func": recovery_copy.autoencoder,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },
    # "lightgbm_v2": {
    #     "func": recovery_copy.lightgbm_v2,
    #     "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    # },
    "dlinear": {
        "func": recovery_copy.dlinear,
        "args": (train, val, op_sales_masked_train, op_sales_masked_val, outside_slice_train, outside_slice_val),
    },          
    #=== LightGBM Recovery v2 Finished ===
    # Imputed 14,311,536 hourly cells
    # Mean raw sale_amount: 0.9986
    # Mean recovered sales: 1.1762
    # Total runtime: 803.53 seconds
    # Gespeichert: [2.3159428  0.55325204 5.3        ... 3.8        2.2        2.1       ]
    # Verarbeitungszeit:  0:13:23.918970
}

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
    recovery_column = f"recovered_daily_sales_{recovery_name}"

    current_time = datetime.now()

    print(f"\n=== Running recovery method: {recovery_name} at {current_time} ===")
    recovered_daily = method["func"](*method["args"])
    history[recovery_column] = recovered_daily
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {recovered_daily.mean():.4f}")

    recovery_folder = "recovered_column"
    np.save(f"{recovery_folder}/{recovery_column}.npy", recovered_daily)

    print("Gespeichert:", recovered_daily)

    processing_time = datetime.now()-current_time

    print("Verarbeitungszeit: ", processing_time)

    with open(f"{recovery_folder}/recovery_processing_time.json", "r") as f:
        content = f.read()
        time = json.loads(content) if content.strip() else {}

    time[recovery_column] = processing_time.total_seconds()

    with open(f"{recovery_folder}/recovery_processing_time.json", "w") as f:
        json.dump(time, f)

