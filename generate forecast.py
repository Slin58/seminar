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

# ------------------------------------------------------------
# 1. Alle Recovery-Werte laden
# ------------------------------------------------------------
folder = "recovered_column"

for datei in os.listdir(folder):
    if datei.endswith(".npy"):
        print("Loaded", datei.replace(".npy", "").replace("recovered_daily_sales_", ""))
        path = os.path.join(folder, datei)

        loaded_arr = np.load(path)

        history[datei.replace(".npy", "")] = loaded_arr

# ------------------------------------------------------------
# 2. Train/Validation Split neu machen
# ------------------------------------------------------------
train_r, val_r = utils.time_split(history, horizon=7)

# ------------------------------------------------------------
# 3. Forecast-Modelle definieren
# ------------------------------------------------------------
print("Forecasting")

# TODO start for all the columns that got overwritten in forecast_processing_time

forecast_models = {
    #"global_mean_forecast": forecast.global_mean,
    #"seasonal_naive_forecast": forecast.seasonal_naive,
    #"rolling_28d_forecast": forecast.rolling_28d,
    #"single_exponential_smoothing": forecast.single_exponential_smoothing,
    #"double_exponential_smoothing": forecast.double_exponential_smoothing,
    #"triple_exponential_smoothing": forecast.triple_exponential_smoothing,
    #"simple_exponential_smoothing": forecast.simple_exponential_smoothing,
    #"holt_winters_exp_forecast": forecast.holt_winters_exp_forecast,
    #"exponential_smoothing_forecast": forecast.exponential_smoothing,
    #"arima_forecast": forecast.arima, # TODO lädt zu lange -> optimieren
    #"lightgbm_forecast": forecast.lightgbm_forecast,
    #"xgboost_forecast": forecast.xgboost_forecast,
    #"random_forest_forecast": forecast.random_forest_forecast,
    #"random_forest_forecast_optimized": forecast.random_forest_forecast_optimized,
    ##"random_forest_forecast_feature_optimized": forecast.random_forest_forecast_feature_optimized, # TODO dauert in raw_sales 10 min -> ca 4 h Ladezeit
    #"cnn_forecast": forecast.cnn_forecast, # TODO Training samples: 2,750,000
        # Window: 28
        # Epochs: 5
        # Epoch 1/5 - Loss: 0.241519
        # Epoch 2/5 - Loss: 0.202200
        # Epoch 3/5 - Loss: 0.189820
        # Epoch 4/5 - Loss: 0.186992
        # Epoch 5/5 - Loss: 0.183760
        # Finished: raw_sales + cnn_forecast (0:27:36.176654)
    #"cnn_forecast_fast": forecast.cnn_forecast_fast, # TODO Training samples: 3,100,000
        # Window: 21
        # Epochs: 3
        # Batch size: 16384
        # Epoch 1/3 - Loss: 0.301544
        # Epoch 2/3 - Loss: 0.232282
        # Epoch 3/3 - Loss: 0.226754
        # Finished: raw_sales + cnn_forecast_fast (0:14:35.852208)
    #"cnn_forecast_balanced": forecast.cnn_forecast_balanced, # TODO Training samples: 2,750,000
        # Window: 28
        # Epochs: 3
        # Batch size: 16384
        # Epoch 1/3 - Loss: 0.318309
        # Epoch 2/3 - Loss: 0.230634
        # Epoch 3/3 - Loss: 0.213261
        # Finished: raw_sales + cnn_forecast_balanced (0:15:26.532638)
    #"cnn_forecast_3epochs": forecast.cnn_forecast_3epochs, # TODO Training samples: 2,750,000
        # Window: 28
        # Epochs: 3
        # Batch size: 16384
        # Epoch 1/3 - Loss: 0.291431
        # Epoch 2/3 - Loss: 0.226947
        # Epoch 3/3 - Loss: 0.216623
        # Finished: raw_sales + cnn_forecast_3epochs (0:16:38.939091)
    #"cnn_forecast_original_3epochs": forecast.cnn_forecast_original_3epochs, # TODO Training samples: 2,750,000
        # Window: 28
        # Epochs: 3
        # Epoch 1/3 - Loss: 0.241519
        # Epoch 2/3 - Loss: 0.202200
        # Epoch 3/3 - Loss: 0.189820
        # Finished: raw_sales + cnn_forecast_original_3epochs (0:16:49.105764)
    #"lstm_forecast": forecast.lstm_forecast, # TODO Training samples: 2,750,000
        # Window: 28
        # Epochs: 3
        # Batch size: 4096
        # Epoch 1/3 - Loss: 0.265970
        # Epoch 2/3 - Loss: 0.184966
        # Epoch 3/3 - Loss: 0.168801
        # Finished: raw_sales + lstm_forecast (0:25:00.256761)
    #"lstm_forecast_fast": forecast.lstm_forecast_fast, # TODO Training samples: 3,100,000
        # Window: 21
        # Epochs: 3
        # Batch size: 8192
        # Epoch 1/3 - Loss: 0.512266
        # Epoch 2/3 - Loss: 0.210390
        # Epoch 3/3 - Loss: 0.195152
        # Predicting validation...
        # Finished: raw_sales + lstm_forecast_fast (0:05:45.542621)
    #"catboost_forecast": forecast.catboost_forecast, # TODO Run: raw_sales + catboost_forecast
        # Finished: raw_sales + catboost_forecast (0:26:24.784278)
    #"catboost_forecast_fast": forecast.catboost_forecast_fast,
    #"catboost_forecast_optimized": forecast.catboost_forecast_optimized, # TODO Run: raw_sales + catboost_forecast_optimized
        # Finished: raw_sales + catboost_forecast_optimized (0:29:31.443697)
    #"lightgbm_forecast_optimized": forecast.lightgbm_forecast_optimized,
    #"lightgbm_forecast_feature_optimized": forecast.lightgbm_forecast_feature_optimized,
    "xgboost_forecast_feature_optimized": forecast.xgboost_forecast_feature_optimized, # TODO Run: raw_sales + xgboost_forecast_feature_optimized
        # Finished: raw_sales + xgboost_forecast_feature_optimized (0:30:48.924779)
    }

# ------------------------------------------------------------
# 4. Forecasts zu allen Recoevery Werten ausführen
# ------------------------------------------------------------
all_results = {}

with open("recovered_column/results.json", "r") as f:
    content = f.read()
    all_results = json.loads(content) if content.strip() else {}

for forecast_name, forecast_func in forecast_models.items():
    processing_time = datetime.now() - datetime.now()
    counter = 0

    # for col in history.columns:
    #     if col.startswith("recovered_daily_sales_") or col == "sale_amount":
    #             current_time = datetime.now()

    #             val_pred = forecast_func(
    #                 train_df=train_r,
    #                 val_df=val_r,
    #                 target_col=col
    #             )
        
    #             name = col.replace("recovered_daily_sales_", "")
    #             if name == "sale_amount":
    #                 name = "raw_sales"

    #             result_name = f"{name} + {forecast_name}"
    #             all_results[result_name] = utils.evaluate_forecast(val_pred)
                
    #             processing_time += datetime.now()-current_time
    #             counter += 1

    for col in history.columns:
        if col.startswith("recovered_daily_sales_") or col == "sale_amount":

            name = col.replace("recovered_daily_sales_", "")
            if name == "sale_amount":
                name = "raw_sales"

            result_name = f"{name} + {forecast_name}"

            # pred_folder = "forecast_predictions" # TODO
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

            val_pred = forecast_func(
                train_df=train_r,
                val_df=val_r,
                target_col=col
            )

            # ------------------------------------------------------------ 
            # Prediction speichern für spätere Ensembles
            # ------------------------------------------------------------
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
# ------------------------------------------------------------
# 5. Ergebnisse anzeigen
# ------------------------------------------------------------
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