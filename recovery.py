# recovery.py contains all recovery methods we implemented
# recovery methods need to add column: "recovered_daily_sales" to history

# TODO bei recovery: recovern von gleichen city_id, store_id, management_group_id, first_category_id, second_category_id, third_category_id, product_id ??

import numpy as np

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
    print(f"Mean recovered sales: {history['recovered_daily_sales'].mean():.4f}")

def global_mean(history): # globaler Durschnitt - Laura
    return

def per_series_mean(history): # Durchschnitt derselben series_id - Nils # TODO  & derselben Stunde
    recovered_daily = history["sale_amount"].where(history["is_censored"] == 0, np.nan)

    series_mean = recovered_daily.groupby(history["series_id"]).transform("mean")

    recovered_daily = recovered_daily.fillna(series_mean)

    history["recovered_daily_sales"] = recovered_daily

    print(f"Mean raw sale_amount: {history['sale_amount'].mean():.4f}")
    print(f"Mean recovered sales: {history['recovered_daily_sales'].mean():.4f}")


def weekday_mean(history): # Durchschnitt der gleichen Wochentage - Laura
    return

def hourly_mean(history): # Durchschnitt der gleichen Stunde - Nils
    return


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

