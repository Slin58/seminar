# recovery.py contains all recovery methods we implemented
# recovery methods need to add column: "recovered_daily_sales" to history

# TODO bei recovery: recovern von gleichen city_id, store_id, management_group_id, first_category_id, second_category_id, third_category_id, product_id ??

import numpy as np
import pandas as pd

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

def global_mean(history, op_stock, op_sales, op_sales_masked, outside_slice):  # globaler Durchschnitt - TODO übergebene Variabeln anpassen
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

# Seasonal imputation: - Nils # TODO ist das nicht das gleiche, wie weekday_mean?
def seasonal_mean(history):  # Durchschnitt desselben Wochentags - Nils
    recovered_daily = history["sale_amount"].where(history["is_censored"] == 0, np.nan)

    dayofweek = pd.to_datetime(history["dt"]).dt.dayofweek
    hour = pd.to_datetime(history["dt"]).dt.hour

    seasonal_key = dayofweek.astype(str) + "_" + hour.astype(str)
    seasonal_mean = recovered_daily.groupby(seasonal_key).transform("mean")

    recovered_daily = recovered_daily.fillna(seasonal_mean)

    history["recovered_daily_sales_seasonal_mean"] = recovered_daily

    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_seasonal_mean'].mean():.4f}")

def hour_per_seasonal_mean(history, op_sales_masked, outside_slice):  # Durchschnitt desselben Wochentags & derselben Stunde - Nils
    imputed = op_sales_masked.copy()

    # ---------- SEASONAL KEYS (weekday x hour) ----------
    dayofweek = pd.to_datetime(history["dt"]).dt.dayofweek.values  # 0–6
    n_rows, n_hours = imputed.shape

    # seasonal_code = weekday * n_hours + hour_index → unique key per (weekday, hour) combo
    hour_indices = np.arange(n_hours)  # 0..n_hours-1
    seasonal_codes_2d = dayofweek[:, None] * n_hours + hour_indices[None, :]  # (n_rows, n_hours)
    n_seasonal = 7 * n_hours

    # ---------- GLOBAL HOURLY MEANS (fallback) ----------
    global_hourly_means = np.nanmean(imputed, axis=0)

    # ---------- COMPUTE PER-SEASONAL MEANS ----------
    seasonal_means_flat = np.full(n_seasonal, np.nan)

    for h in range(n_hours):
        values = imputed[:, h]
        valid_mask = ~np.isnan(values)
        codes = seasonal_codes_2d[:, h]  # weekday * n_hours + h

        sums = np.bincount(codes[valid_mask], weights=values[valid_mask], minlength=n_seasonal)
        counts = np.bincount(codes[valid_mask], minlength=n_seasonal)

        means = sums / np.maximum(counts, 1)
        means[counts == 0] = global_hourly_means[h]  # fallback

        # only write the codes that appear in this hour column
        seasonal_means_flat[codes] = means[codes]

    # ---------- IMPUTE ----------
    nan_mask = np.isnan(imputed)

    replacement_values = seasonal_means_flat[seasonal_codes_2d]  # (n_rows, n_hours)

    imputed[nan_mask] = replacement_values[nan_mask]

    imputed = np.maximum(imputed, 0)

    imputed_count = nan_mask.sum()

    # ---------- REBUILD DAILY SALES ----------
    recovered_sum = np.nansum(imputed, axis=1)

    recovered_daily = recovered_sum + outside_slice

    history["recovered_daily_sales_hour_per_seasonal_mean"] = recovered_daily

    print(f"Imputed {imputed_count:,} hourly cells")
    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales_hour_per_seasonal_mean'].mean():.4f}")

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


def interpolation_spline(history, op_sales_masked, outside_slice, order=3):  # Spline-Interpolation - Laura

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

def interpolation_polynomial(history, op_sales_masked, outside_slice, order=2):  # Polynomial-Interpolation - Laura

    imputed = op_sales_masked.copy()

    imputed_count = np.isnan(imputed).sum()

    n_hours = imputed.shape[1]

    # Jede Stunde einzeln
    for h in range(n_hours):

        s = pd.Series(imputed[:, h])

        # Polynomial braucht genug bekannte Werte
        if s.notna().sum() > order:

            interpolated = s.interpolate(
                method="polynomial",
                order=order,
                limit_direction="both"
            )

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

def kalman_smoothing(history, op_sales_masked, outside_slice):  # Kalman Smoothing / State Space - Laura TODO lädt über 9 min (nicht fertig laden lassen) - Laura

    from statsmodels.tsa.statespace.structural import UnobservedComponents

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
                model = UnobservedComponents(
                    s,
                    level="local level"
                )

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

def stl_real(history, op_sales_masked, outside_slice, period=7): # TODO lädt über 5 min (nicht fertig laden lassen) - Laura
    from statsmodels.tsa.seasonal import STL

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
            stl = STL(
                s_filled,
                period=period,
                robust=True
            )

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

def knn(history, op_sales_masked, outside_slice, n_neighbors=5):  # KNN-Imputation - Laura

    from sklearn.impute import KNNImputer

    imputed = op_sales_masked.copy()

    imputed_count = np.isnan(imputed).sum()

    # KNNImputer arbeitet spaltenweise über die 16 Stunden
    imputer = KNNImputer(
        n_neighbors=n_neighbors,
        weights="distance"
    )

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

def random_forest(history): # Random Forest / XGBoost basierte Imputation
    return

def iterative(history): # Iterative Imputation (z.B. MICE)
    return


# Spezifische Demandrevovery Modelle: - Nils

def lost_sales_model(history): # was ist das? Laut ChatGPT ein Überbegriff für Modelle, die versuchen verlorene Umsätze zu schätzen, z.B. mit Random Forest oder XGBoost. 
    return

# test letztes von Claude AI
def tobit_model(history):
    from scipy.optimize import minimize
    from scipy.special import ndtr, log_ndtr

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
    result = minimize(
        neg_log_likelihood,
        np.zeros(n_features + 1, dtype=np.float64),
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    beta_hat  = result.x[:-1]
    sigma_hat = float(np.exp(result.x[-1]))

    # ---------- PREDICT ----------
    mu_hat   = (X @ beta_hat).ravel()
    alpha    = mu_hat / sigma_hat
    pdf_a    = np.exp(-0.5 * alpha * alpha) / np.sqrt(2 * np.pi)
    cdf_a    = ndtr(alpha)
    lambda_  = pdf_a / np.maximum(cdf_a, 1e-12)

    e_y_star = mu_hat + sigma_hat * lambda_
    history["recovered_daily_sales_tobit"] = np.where(
        is_censored, np.maximum(e_y_star, 0), y
    )

    print(f"Converged: {result.success} | {result.message}")
    print(f"sigma_hat: {sigma_hat:.4f}")
    print(f"Mean raw sale_amount:  {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales:  {history['recovered_daily_sales_tobit'].mean():.4f}")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr, log_ndtr
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# ── per-series worker (must be top-level for pickling) ──────────────────────
def _fit_one_series(args):
    (store_id, product_id), df = args

    df = df.reset_index(drop=True)
    n  = len(df)

    # need enough obs to identify the model
    if n < 15:
        return store_id, product_id, None, "too_few_rows"

    # ── features ────────────────────────────────────────────────────────────
    hours_matrix = np.vstack(df["hours_sale"].values)
    hours_stock  = np.vstack(df["hours_stock_status"].values)
    peak         = slice(9, 21)

    hours_features = np.column_stack([
        hours_matrix[:, peak].sum(axis=1),
        hours_matrix[:, :9].sum(axis=1) + hours_matrix[:, 21:].sum(axis=1),
        hours_stock[:, peak].sum(axis=1),
        np.clip(1 - df["stock_hour6_22_cnt"].values / 16, 1e-3, 1 - 1e-3),
    ])

    dt = pd.to_datetime(df["dt"])
    base = np.column_stack([
        dt.dt.dayofweek.values,
        df["avg_temperature"].fillna(0).values,
        df["avg_humidity"].fillna(0).values,
        df["avg_wind_level"].fillna(0).values,
        df["precpt"].fillna(0).values,
        df["holiday_flag"].values,
        df["activity_flag"].values,
        df["discount"].values,
        np.ones(n),
    ])

    X           = np.hstack([base, hours_features]).astype(np.float64)
    y           = df["sale_amount"].values.astype(np.float64)
    is_censored = df["is_censored"].values.astype(bool)
    obs_mask    = ~is_censored

    if obs_mask.sum() < 5:
        return store_id, product_id, None, "too_few_observed"

    X_obs, y_obs       = X[obs_mask], y[obs_mask]
    X_cen              = X[is_censored]
    avail_frac_cen     = hours_features[is_censored, 3]
    cen_weight         = (1 - avail_frac_cen)

    LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)

    def neg_ll(params):
        beta, log_sigma = params[:-1], params[-1]
        sigma = np.exp(log_sigma)

        z      = (y_obs - X_obs @ beta) / sigma
        ll_obs = (-log_sigma - LOG_SQRT_2PI - 0.5 * z * z).sum()

        if X_cen.shape[0] > 0:
            ll_cen = (log_ndtr(-(X_cen @ beta) / sigma) * cen_weight).sum()
        else:
            ll_cen = 0.0

        return -(ll_obs + ll_cen)

    result = minimize(
        neg_ll,
        np.zeros(X.shape[1] + 1),
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    beta_hat  = result.x[:-1]
    sigma_hat = np.exp(result.x[-1])

    mu      = X @ beta_hat
    alpha   = mu / sigma_hat
    lambda_ = np.exp(-0.5 * alpha**2) / (np.sqrt(2 * np.pi) * np.maximum(ndtr(alpha), 1e-12))

    recovered = np.where(is_censored, np.maximum(mu + sigma_hat * lambda_, 0), y)

    out = df[["store_id", "product_id", "dt"]].copy()
    out["recovered_daily_sales_tobit"] = recovered
    out["converged"]                   = result.success

    return store_id, product_id, out, "ok"


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

