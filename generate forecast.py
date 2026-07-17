import numpy as np
import utils
from datasets import load_dataset
import forecast
import json
import os
from datetime import datetime

ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K")

# Data preparation
train_raw = ds["train"].to_pandas()
eval_raw = ds["eval"].to_pandas()

history = utils.prepare_panel(train_raw)
history = utils.flag_censoring(history)
history = utils.make_features(history)

# 1. Train split
train_r, val = utils.time_split(history, horizon=7)


# 2. Alle Recovery-Werte laden
folder = "recovered_column"

for datei in os.listdir(folder):
    if datei.endswith(".npy"):
        print("Loaded", datei.replace(".npy", "").replace("recovered_daily_sales_", ""))
        path = os.path.join(folder, datei)

        loaded_arr = np.load(path)

        train_r[datei.replace(".npy", "")] = loaded_arr

# 2. Train/Validation Split neu machen
#train_r, val_r = utils.time_split(history, horizon=7)

# 3. Forecast-Modelle definieren
print("Forecasting")

forecast_models = {
    "global_mean_forecast": forecast.global_mean,
    "seasonal_naive_forecast": forecast.seasonal_naive,
    "rolling_28d_forecast": forecast.rolling_28d,
    #"single_exponential_smoothing": forecast.single_exponential_smoothing,
    #"double_exponential_smoothing": forecast.double_exponential_smoothing,
    #"triple_exponential_smoothing": forecast.triple_exponential_smoothing,
    #"simple_exponential_smoothing": forecast.simple_exponential_smoothing,
    #"holt_winters_exp_forecast": forecast.holt_winters_exp_forecast,
    #"exponential_smoothing_forecast": forecast.exponential_smoothing,
    #"arima_forecast": forecast.arima, # 
    #"arima_like_fast": forecast.arima_like_fast,
    #"arima_like_fast_vectorized": forecast.arima_like_fast_vectorized,
    #"lightgbm_forecast": forecast.lightgbm_forecast,
    "xgboost_forecast": forecast.xgboost_forecast,
    #"random_forest_forecast": forecast.random_forest_forecast,
    #"random_forest_forecast_optimized": forecast.random_forest_forecast_optimized,
    #"random_forest_forecast_feature_optimized": forecast.random_forest_forecast_feature_optimized,
    #"cnn_forecast": forecast.cnn_forecast,
    #"cnn_forecast_fast": forecast.cnn_forecast_fast,
    #"cnn_forecast_balanced": forecast.cnn_forecast_balanced,
    #"cnn_forecast_3epochs": forecast.cnn_forecast_3epochs,
    #"cnn_forecast_original_3epochs": forecast.cnn_forecast_original_3epochs,
    #"lstm_forecast": forecast.lstm_forecast,
    #"lstm_forecast_fast": forecast.lstm_forecast_fast, 
    #"catboost_forecast": forecast.catboost_forecast, # TODO Potenzial
    #"catboost_forecast_fast": forecast.catboost_forecast_fast,
    #"catboost_forecast_optimized": forecast.catboost_forecast_optimized,
    #"catboost_forecast_optimized_v2": forecast.catboost_forecast_optimized_v2, # lädt zu lange
    #"catboost_forecast_fast_numeric_v2": forecast.catboost_forecast_fast_numeric_v2,
    #"lightgbm_forecast_optimized": forecast.lightgbm_forecast_optimized,
    #"lightgbm_forecast_feature_optimized": forecast.lightgbm_forecast_feature_optimized, # TODO Potenzial
    #"lightgbm_forecast_feature_optimized_v2": forecast.lightgbm_forecast_feature_optimized_v2,
    "lightgbm_forecast_feature_optimized_v3": forecast.lightgbm_forecast_feature_optimized_v3, # bestes
    #"lightgbm_forecast_feature_optimized_v4": forecast.lightgbm_forecast_feature_optimized_v4,
    #"lightgbm_forecast_feature_optimized_v5": forecast.lightgbm_forecast_feature_optimized_v5,
    #"lightgbm_forecast_feature_optimized_v6_feature_selection": forecast.lightgbm_forecast_feature_optimized_v6_feature_selection,
        # infos siehe dokument
    #"xgboost_forecast_feature_optimized": forecast.xgboost_forecast_feature_optimized, # TODO Potenzial
    #"xgboost_forecast_feature_fast": forecast.xgboost_forecast_feature_fast,
    #"hist_gradient_boosting_forecast": forecast.hist_gradient_boosting_forecast,
    #"hist_gradient_boosting_forecast_optimized": forecast.hist_gradient_boosting_forecast_optimized, # TODO Potenzial
    #"extra_trees_forecast": forecast.extra_trees_forecast,
    #"dlinear_forecast": forecast.dlinear
    }

recovery_models = [] # if list empty all recovery methods are used
#recovery_models = ["recovered_daily_sales_stl_real", "sale_amount", "recovered_daily_sales_lightgbm_v2", "recovered_daily_sales_lightgbm", "recovered_daily_sales_dlinear"] # if list empty all recovery methods are used

# 4. Forecasts zu allen Recovery Werten ausführen
all_results = {}

with open("recovered_column/results.json", "r") as f:
    content = f.read()
    all_results = json.loads(content) if content.strip() else {}

for forecast_name, forecast_func in forecast_models.items():
    print(f"starting at: {datetime.now()}")
    processing_time = datetime.now() - datetime.now()
    counter = 0

    for col in train_r.columns:
        if (col.startswith("recovered_daily_sales_") or col == "sale_amount") and (col in recovery_models or len(recovery_models) == 0):

            name = col.replace("recovered_daily_sales_", "")
            if name == "sale_amount":
                name = "raw_sales"

            # TODO nicht überschreiben von vorhandenen Werten, nur wenn nicht vorhanden

            result_name = f"{name} + {forecast_name}"

            # pred_folder = "forecast_predictions" # TODO Ensembles
            # os.makedirs(pred_folder, exist_ok=True)

            # pred_name = result_name.replace(" ", "_").replace("+", "plus").replace("/", "_")
            # pred_path = f"{pred_folder}/{pred_name}.npy"

            # if result_name in all_results and os.path.exists(pred_path):
            #     print(f"Skip: {result_name}")
            #     continue


            # Bereits vorhanden -> überspringen
            if result_name in all_results:
                print(f"Skip: {result_name}")
                continue

            print(f"Run: {result_name}")

            current_time = datetime.now()

            val_pred = forecast_func(train_df=train_r, val_df=val, recovery_col=col) 
            
            # Prediction speichern für spätere Ensembles:
            # 
            # pred_folder = "forecast_predictions"
            # os.makedirs(pred_folder, exist_ok=True)

            # pred_name = result_name.replace(" ", "_").replace("+", "plus").replace("/", "_")

            # np.save(
            #     f"{pred_folder}/{pred_name}.npy",
            #     val_pred["prediction"].to_numpy()
            # )

            all_results[result_name] = utils.evaluate_forecast(val_pred)

            processing_time += datetime.now() - current_time
            counter += 1

            print(f"Finished: {result_name} ({datetime.now() - current_time})")

    if counter > 0:
        with open("recovered_column/forecast_processing_time.json", "r") as f:
            content = f.read()
            time = json.loads(content) if content.strip() else {}

        time[forecast_name] = processing_time.total_seconds() / counter
        with open("recovered_column/forecast_processing_time.json", "w") as f:
            json.dump(time, f) 


with open("recovered_column/results.json", "w") as f:
    json.dump(all_results, f)



"""
# 5. Ergebnisse anzeigen
all_results_df = pd.DataFrame(all_results).T


print("=== Dynamic Recovery + Forecast Results ===")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
display(all_results_df.sort_values("harmonic_mean"))

print("=== Processing time recovery methods ===")
with open("recovered_column/processing_time.json", "r") as f:
    content = f.read()
    time = json.loads(content) if content.strip() else {}

rows = []
for key, seconds in time.items():
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    ms = int((seconds % 1) * 1000)
    name = key.removeprefix("recovered_daily_sales_")
    rows.append({"method": name, "time": f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}", "seconds": seconds})

df = pd.DataFrame(rows).sort_values("seconds").drop(columns="seconds").reset_index(drop=True)
df.index += 1

display(df)
"""