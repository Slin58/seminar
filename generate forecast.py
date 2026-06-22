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

forecast_models = {
    "global_mean_forecast": forecast.global_mean,
    "seasonal_naive_forecast": forecast.seasonal_naive,
    "rolling_28d_forecast": forecast.rolling_28d,
    "simple_exponential_smoothing": forecast.simple_exponential_smoothing,
    #"exponential_smoothing_forecast": forecast.exponential_smoothing,
    #"arima_forecast": forecast.arima,
    "lightgbm_forecast": forecast.lightgbm_forecast,
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

            all_results[result_name] = utils.evaluate_forecast(val_pred)

            processing_time += datetime.now() - current_time
            counter += 1

            print(
                f"Finished: {result_name} "
                f"({datetime.now() - current_time})"
            )


    with open("recovered_column/forecast_processing_time.json", "r") as f:
        content = f.read()
        time = json.loads(content) if content.strip() else {}
    time[forecast_name] = processing_time.total_seconds()
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