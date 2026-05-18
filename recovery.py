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

    history["recovered_daily_sales"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_random_sampling'].mean():.4f}")

def global_mean(history): # globaler Durschnitt - Laura
    return

def per_series_mean(history): # Durchschnitt derselben series_id - Nils
    recovered_daily = history["sale_amount"].where(history["is_censored"] == 0, np.nan)

    series_mean = recovered_daily.groupby(history["series_id"]).transform("mean")

    recovered_daily = recovered_daily.fillna(series_mean)

    history["recovered_daily_sales"] = recovered_daily

    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_per_series_mean'].mean():.4f}")

def hour_per_series_mean(history, op_sales_masked, hours_sale_with_stockout): # Durchschnitt derselben series_id & derselben Stunde - Nils

    imputed = op_sales_masked.copy()
    imputed_count = 0

    # unique series
    unique_series = history["series_id"].unique()

    # process each series separately
    for sid in unique_series:
        # rows belonging to this series
        series_mask = history["series_id"].values == sid

        # hourly matrix for this series
        series_data = imputed[series_mask]

        # calculate hourly mean ignoring NaNs
        hourly_means = np.nanmean(series_data, axis=0)

        # fallback if an entire hour is NaN
        global_hourly_means = np.nanmean(imputed, axis=0)
        hourly_means = np.where(
            np.isnan(hourly_means),
            global_hourly_means,
            hourly_means
        )

        # impute hour by hour
        for h in range(16):

            col = series_data[:, h]

            # missing values
            missing_mask = np.isnan(col)
            n_miss = missing_mask.sum()

            if n_miss > 0:
                col[missing_mask] = np.maximum(0, hourly_means[h])
                imputed_count += n_miss

            series_data[:, h] = col

        imputed[series_mask] = series_data

    recovered_sum = np.nansum(imputed, axis=1)
    history["recovered_daily_sales"] = recovered_sum - hours_sale_with_stockout

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales'].mean():.4f}")


def hour_per_series_mean_fast(history, op_sales_masked, hours_sale_with_stockout):
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

    history["recovered_daily_sales"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales'].mean():.4f}")

def weekday_mean(history): # Durchschnitt der gleichen Wochentage - Laura
    return

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

    history["recovered_daily_sales"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales'].mean():.4f}")


# Moving averages: - Laura

def rolling_mean(history): # simple_moving_average
    return

def exponential_moving_average(history): # ema
    return


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

