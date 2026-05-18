# recovery.py contains all recovery methods we implemented
# recovery methods need to add column: "recovered_daily_sales" to history

# TODO bei recovery: recovern von gleichen city_id, store_id, management_group_id, first_category_id, second_category_id, third_category_id, product_id ??

import numpy as np
import pandas as pd

# Einfache Imputation Methoden:
def random_sampling(history, op_sales_masked, hours_sale_with_stockout, rng): # Simple recovery: random pool sampling
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
    recovered_daily = recovered_sum - hours_sale_with_stockout

    history["recovered_daily_sales_random_sampling"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_random_sampling'].mean():.4f}")

def global_mean(history, op_stock, op_sales, op_sales_masked):  # globaler Durchschnitt - Laura
    visible_sum = np.nansum(
        np.where(op_stock == 0, op_sales, 0),
        axis=1
    )
    imputed = op_sales_masked.copy()
    # Durchschnitt über alle sichtbaren Stundenwerte
    mean_value = np.nanmean(imputed)
    # Nur NaN-Werte ersetzen
    mask = np.isnan(imputed)
    imputed_count = mask.sum()
    imputed[mask] = mean_value
    # Rebuild corrected daily target
    recovered_sum = np.nansum(imputed, axis=1)
    outside_slice = np.maximum(
        history["sale_amount"].values.astype(np.float32) - visible_sum,
        0
    )
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

def hour_per_series_mean(history, op_sales_masked, hours_sale_with_stockout): # Durchschnitt derselben series_id & derselben Stunde - Nils
    imputed = op_sales_masked.copy()

    # ---------- SERIES IDS ----------
    series_codes, unique_series = pd.factorize(history["series_id"], sort=False)
    n_series = len(unique_series)

    n_rows, n_hours = imputed.shape

    # ---------- GLOBAL HOURLY MEANS ----------
    global_hourly_means = np.nanmean(imputed, axis=0)

    # ---------- COMPUTE PER-SERIES HOURLY MEANS ----------
    # result shape: (n_series, n_hours)
    series_hourly_means = np.full((n_series, n_hours), np.nan)

    for h in range(n_hours):

        values = imputed[:, h]

        valid_mask = ~np.isnan(values)

        # sums per series
        sums = np.bincount(series_codes[valid_mask], weights=values[valid_mask], minlength=n_series)

        # counts per series
        counts = np.bincount(series_codes[valid_mask], minlength=n_series)

        means = sums / np.maximum(counts, 1)

        # fallback to global mean if no observations
        means[counts == 0] = global_hourly_means[h]

        series_hourly_means[:, h] = means

    # ---------- IMPUTE ----------
    nan_mask = np.isnan(imputed)

    # build replacement matrix directly
    replacement_values = series_hourly_means[series_codes]

    imputed[nan_mask] = replacement_values[nan_mask]

    imputed = np.maximum(imputed, 0)

    imputed_count = nan_mask.sum()

    # ---------- REBUILD DAILY SALES ----------
    recovered_sum = np.nansum(imputed, axis=1)

    recovered_daily = recovered_sum - hours_sale_with_stockout

    history["recovered_daily_sales_hour_per_series_mean"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_hour_per_series_mean'].mean():.4f}")

def weekday_per_series_mean():
    return

def weekday_mean(history, op_stock, op_sales, op_sales_masked):  # Durchschnitt gleicher Wochentage - Laura
    # Wochentag hinzufügen
    history["weekday"] = history["dt"].dt.weekday
    visible_sum = np.nansum(
        np.where(op_stock == 0, op_sales, 0),
        axis=1
    )
    imputed = op_sales_masked.copy()
    imputed_count = 0
    # Jede Stunde einzeln
    for h in range(16):
        # Alle Werte dieser Stunde
        col = imputed[:, h]
        # Fehlende Werte finden
        mask = np.isnan(col)
        # Für jede fehlende Stelle
        missing_idx = np.where(mask)[0]
        for idx in missing_idx:
            # Wochentag dieser Zeile
            wd = history.iloc[idx]["weekday"]
            # Alle sichtbaren Werte gleicher Wochentage
            same_weekday_mask = (
                (history["weekday"] == wd).values &
                (~np.isnan(col))
            )
            pool = col[same_weekday_mask]
            # Falls Werte existieren
            if len(pool) > 0:
                mean_value = np.mean(pool)
            else:
                mean_value = np.nanmean(col)
            imputed[idx, h] = mean_value
            imputed_count += 1
    # Rebuild corrected daily target
    recovered_sum = np.nansum(imputed, axis=1)
    outside_slice = np.maximum(
        history["sale_amount"].values.astype(np.float32) - visible_sum,
        0
    )
    recovered_daily = outside_slice + recovered_sum
    history["recovered_daily_sales_weekday_mean"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_weekday_mean'].mean():.4f}")

def hourly_mean(history, op_sales_masked, hours_sale_with_stockout): # Durchschnitt der gleichen Stunde - Nils
    imputed = op_sales_masked.copy()

    global_hourly_means = np.nanmean(imputed, axis=0)

    nan_mask = np.isnan(imputed)

    replacement_values = np.tile(global_hourly_means, (imputed.shape[0], 1))

    imputed[nan_mask] = replacement_values[nan_mask]

    imputed = np.maximum(imputed, 0)

    imputed_count = nan_mask.sum()

    recovered_sum = np.nansum(imputed, axis=1)

    recovered_daily = recovered_sum - hours_sale_with_stockout

    history["recovered_daily_sales_hourly_mean"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_hourly_mean'].mean():.4f}")


# Moving averages: - Laura
def rolling_mean(history, op_stock, op_sales, op_sales_masked, window=7):  # SMA / Rolling Mean - Laura
    visible_sum = np.nansum(
        np.where(op_stock == 0, op_sales, 0),
        axis=1
    )
    imputed = op_sales_masked.copy()
    imputed_count = 0
    # Jede Stunde einzeln
    for h in range(16):
        col = imputed[:, h]
        # Fehlende Werte finden
        mask = np.isnan(col)
        # Positionen der fehlenden Werte
        missing_idx = np.where(mask)[0]
        for idx in missing_idx:
            # Vergangene sichtbare Werte holen
            start = max(0, idx - window)
            previous_values = col[start:idx]
            # Nur sichtbare Werte behalten
            previous_values = previous_values[~np.isnan(previous_values)]
            # Falls Werte existieren
            if len(previous_values) > 0:
                mean_value = np.mean(previous_values)
            else:
                # Fallback falls keine Werte existieren
                mean_value = np.nanmean(col)
            imputed[idx, h] = mean_value
            imputed_count += 1
    # Rebuild corrected daily target
    recovered_sum = np.nansum(imputed, axis=1)
    outside_slice = np.maximum(
        history["sale_amount"].values.astype(np.float32) - visible_sum,
        0
    )
    recovered_daily = outside_slice + recovered_sum
    history["recovered_daily_sales_rolling_mean"] = recovered_daily
    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Window size used: {window}")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_rolling_mean'].mean():.4f}")

def exponential_moving_average(history, op_stock, op_sales, op_sales_masked, alpha=0.3):  # EMA - Laura
    visible_sum = np.nansum(
        np.where(op_stock == 0, op_sales, 0),
        axis=1
    )
    imputed = op_sales_masked.copy()
    imputed_count = 0
    # Jede Stunde einzeln
    for h in range(16):
        col = imputed[:, h]
        # Fehlende Werte finden
        mask = np.isnan(col)
        # Positionen fehlender Werte
        missing_idx = np.where(mask)[0]
        for idx in missing_idx:
            # Vergangene sichtbare Werte
            previous_values = col[:idx]
            previous_values = previous_values[
                ~np.isnan(previous_values)
            ]
            # Falls Werte existieren
            if len(previous_values) > 0:
                # EMA berechnen
                ema = previous_values[0]
                for val in previous_values[1:]:
                    ema = alpha * val + (1 - alpha) * ema
                mean_value = ema
            else:
                # Fallback
                mean_value = np.nanmean(col)
            imputed[idx, h] = mean_value
            imputed_count += 1
    # Rebuild corrected daily target
    recovered_sum = np.nansum(imputed, axis=1)
    outside_slice = np.maximum(
        history["sale_amount"].values.astype(np.float32) - visible_sum,
        0
    )
    recovered_daily = outside_slice + recovered_sum
    history["recovered_daily_sales_exponential_moving_average"] = recovered_daily
    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Alpha used: {alpha}")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_exponential_moving_average'].mean():.4f}")

def exponential_moving_average_series(history, op_stock, op_sales, op_sales_masked, alpha=0.3):  # EMA pro series_id - Laura

    visible_sum = np.nansum(
        np.where(op_stock == 0, op_sales, 0),
        axis=1
    )

    imputed = op_sales_masked.copy()

    imputed_count = 0

    # Jede Stunde einzeln
    for h in range(16):

        # Jede Serie einzeln
        for sid in history["series_id"].unique():

            # Zeilen dieser Serie
            series_mask = history["series_id"] == sid

            # Werte dieser Stunde UND dieser Serie
            col = imputed[series_mask, h]

            # Fehlende Werte finden
            mask = np.isnan(col)

            # Positionen fehlender Werte
            missing_idx = np.where(mask)[0]

            for idx in missing_idx:

                # Frühere sichtbare Werte derselben Serie
                previous_values = col[:idx]

                previous_values = previous_values[
                    ~np.isnan(previous_values)
                ]

                # Falls sichtbare Werte existieren
                if len(previous_values) > 0:

                    # EMA berechnen
                    ema = previous_values[0]

                    for val in previous_values[1:]:
                        ema = alpha * val + (1 - alpha) * ema

                    mean_value = ema

                else:
                    # Fallback: Durchschnitt dieser Serie/Stunde
                    mean_value = np.nanmean(col)

                # Fehlenden Wert ersetzen
                col[idx] = mean_value

                imputed_count += 1

            # Zurückschreiben
            imputed[series_mask, h] = col

    # Rebuild corrected daily target
    recovered_sum = np.nansum(imputed, axis=1)

    outside_slice = np.maximum(
        history["sale_amount"].values.astype(np.float32) - visible_sum,
        0
    )

    recovered_daily = outside_slice + recovered_sum

    history["recovered_daily_sales_exponential_moving_average_series"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Alpha used: {alpha}")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_exponential_moving_average_series'].mean():.4f}")


# Seasonal imputation: - Nils

def seasonal_naive(history): # Vorhersage = Wert der Vorwoche (gleicher Wochentag + gleiche Stunde)
    return

# Zeitreihenbasierte Recovery-Methoden: - Laura

def interpolation_linear(): # Interpolieren zwischen zwei Werten (letzter bekannter Wert und nächster bekannter Wert)
    return

def interpolation_spline(history): # Interpolieren zwischen zwei Werten (letzter bekannter Wert und nächster bekannter Wert)
    return

def interpolation_polynomial(history): # Interpolieren zwischen zwei Werten (letzter bekannter Wert und nächster bekannter Wert)
    return

def kalman_smoothing(history): # state space 
    return

def stl_based(history): # zerlegt trend, saison und rest
    return


# ML-basierte Recovery-Methoden: - Laura

def knn(history): # K-nearest neighbors basierte Imputation
    return

def random_forest(history): # Random Forest / XGBoost basierte Imputation
    return

def iterative(history): # Iterative Imputation (z.B. MICE)
    return


# Spezifische Demandrevovery Modelle: - Nils

def lost_sales_model(history):
    return

def tobit_model(history): 
    return

def bayesian_model(history):
    return

def inventory_aware_model(history):
    return


# Deep Learning basierte Recovery-Methoden: - Nils

def autoencoder(history):
    return

def transformer(history): # SAITS, BRITS, GRIN, CSDI
    return

def defusion_model(history):
    return

