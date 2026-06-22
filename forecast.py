
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA

# TODO: Forecast Methoden:

# - random_forest Run: raw_sales + random_forest_forecast
    # [Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
    # [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed: 12.0min
    # [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed: 54.1min
    # [Parallel(n_jobs=-1)]: Done 200 out of 200 | elapsed: 58.4min finished
    # [Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
    # [Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.5s
    # [Parallel(n_jobs=8)]: Done 184 tasks      | elapsed:    2.6s
    # [Parallel(n_jobs=8)]: Done 200 out of 200 | elapsed:    2.8s finished
    # Finished: raw_sales + random_forest_forecast (0:58:52.013636)
# - xgboost
# - lightgbm
# - CNN
# - RNN, LSTM oder Transformer
# - Gaussian Process Regression

# Nils:
# - exponential smoothing (Holt-Winters) pro series_id, mit Fallbacks -> dauert wahrscheinlich zu lange
# - simple tripple exponential smoothing

# Laura: 
# - arima (lädt über eine Stunde nur für raw_sales)
# - sarima

def global_mean(train_df, val_df, target_col):
    """
    Forecast = Durchschnitt der jeweiligen series_id im Training.
    """
    val_pred = val_df.copy()

    series_mean = train_df.groupby("series_id")[target_col].mean()

    val_pred["prediction"] = val_pred["series_id"].map(series_mean)

    fallback = train_df[target_col].mean()
    val_pred["prediction"] = val_pred["prediction"].fillna(fallback)

    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred


def seasonal_naive(train_df, val_df, target_col):
    """
    Forecast = Wert von vor 7 Tagen.
    """
    val_pred = val_df.copy()

    val_start = val_pred["day_idx"].min()

    last_week = train_df[
        train_df["day_idx"].between(val_start - 7, val_start - 1)
    ][["series_id", "day_idx", target_col]].copy()

    last_week["forecast_day"] = last_week["day_idx"] + 7

    last_week = last_week.rename(columns={target_col: "prediction"})

    val_pred = val_pred.merge(
        last_week[["series_id", "forecast_day", "prediction"]],
        left_on=["series_id", "day_idx"],
        right_on=["series_id", "forecast_day"],
        how="left"
    )

    val_pred = val_pred.drop(columns=["forecast_day"], errors="ignore")

    fallback = train_df.groupby("series_id")[target_col].mean()

    val_pred["prediction"] = val_pred["prediction"].fillna(
        val_pred["series_id"].map(fallback)
    )

    global_fallback = train_df[target_col].mean()
    val_pred["prediction"] = val_pred["prediction"].fillna(global_fallback)

    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred


def rolling_28d(train_df, val_df, target_col):
    """
    Forecast = Durchschnitt der letzten 28 Trainingstage pro series_id.
    """
    val_pred = val_df.copy()

    roll28 = train_df.groupby("series_id")[target_col].apply(
        lambda x: x.tail(28).mean()
    )

    val_pred["prediction"] = val_pred["series_id"].map(roll28)

    fallback = train_df.groupby("series_id")[target_col].mean()

    val_pred["prediction"] = val_pred["prediction"].fillna(
        val_pred["series_id"].map(fallback)
    )

    global_fallback = train_df[target_col].mean()
    val_pred["prediction"] = val_pred["prediction"].fillna(global_fallback)

    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred

#TODO single and double exponential smoothing (alpha, beta, gamma einstellen) und holt-winters nochmal optimieren bzw. Chati fragen warum er solange laden
def simple_exponential_smoothing(train_df, val_df, target_col, alpha=0.3):
    """
    Simple Exponential Smoothing (SES).

    Pro series_id wird der exponentiell geglättete Level am Ende der
    Trainingsdaten berechnet:

        S_1 = y_1
        S_t = alpha * y_t + (1 - alpha) * S_{t-1}

    Der finale Level S_T wird als konstanter Forecast für alle
    Validierungstage der jeweiligen series_id verwendet.

    alpha: Glättungsfaktor zwischen 0 und 1.
           - alpha nah an 1 -> reagiert stark auf die letzten Werte
           - alpha nah an 0 -> glättet stark, reagiert langsam
    """
    val_pred = val_df.copy()

    train_sorted = train_df.sort_values(["series_id", "day_idx"])

    levels = (
        train_sorted.groupby("series_id")[target_col]
        .apply(lambda s: s.ewm(alpha=alpha, adjust=False).mean().iloc[-1])
    )
    levels.name = "prediction"

    val_pred = val_pred.merge(
        levels.reset_index(), on="series_id", how="left"
    )

    # Fallback 1: series_id Durchschnitt (falls series_id nicht in train)
    fallback_series = train_df.groupby("series_id")[target_col].mean()
    val_pred["prediction"] = val_pred["prediction"].fillna(
        val_pred["series_id"].map(fallback_series)
    )

    # Fallback 2: globaler Durchschnitt
    global_fallback = train_df[target_col].mean()
    val_pred["prediction"] = val_pred["prediction"].fillna(global_fallback)

    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred


def exponential_smoothing(train_df, val_df, target_col, seasonal_periods=7):
    """
    Forecast via Holt-Winters Exponential Smoothing pro series_id.

    Für jede series_id wird auf den (nach day_idx sortierten)
    Trainingsdaten ein ExponentialSmoothing-Modell gefittet und
    für den benötigten Horizont extrapoliert.

    Fallback-Kette (je nach Datenmenge / Fit-Erfolg):
    1. additiver Trend (gedämpft) + additive Saisonalität (seasonal_periods)
    2. additiver Trend (gedämpft), keine Saisonalität
    3. einfaches Level-Smoothing (SES), kein Trend, keine Saison
    4. series_id Durchschnitt
    5. globaler Durchschnitt
    """
    val_pred = val_df.copy()

    global_fallback = train_df[target_col].mean()
    series_fallback = train_df.groupby("series_id")[target_col].mean()

    predictions = {}

    for series_id, val_group in val_pred.groupby("series_id"):
        train_group = train_df[train_df["series_id"] == series_id].sort_values("day_idx")

        val_days = val_group["day_idx"].sort_values().values

        if train_group.empty:
            fb = global_fallback
            for day_idx in val_days:
                predictions[(series_id, day_idx)] = fb
            continue

        y = train_group[target_col].astype(float).values
        n = len(y)

        train_max_day = train_group["day_idx"].max()
        max_horizon = int(val_days.max() - train_max_day)
        max_horizon = max(max_horizon, 1)

        forecast = None

        # Versuch 1: Trend (gedämpft) + Saisonalität
        if n >= 2 * seasonal_periods:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ExponentialSmoothing(
                        y,
                        trend="add",
                        damped_trend=True,
                        seasonal="add",
                        seasonal_periods=seasonal_periods,
                        initialization_method="estimated",
                    ).fit()
                forecast = model.forecast(max_horizon)
            except Exception:
                forecast = None

        # Versuch 2: nur Trend (gedämpft)
        if forecast is None and n >= 2:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ExponentialSmoothing(
                        y,
                        trend="add",
                        damped_trend=True,
                        seasonal=None,
                        initialization_method="estimated",
                    ).fit()
                forecast = model.forecast(max_horizon)
            except Exception:
                forecast = None

        # Versuch 3: simple exponential smoothing (kein Trend, keine Saison)
        if forecast is None and n >= 1:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ExponentialSmoothing(
                        y,
                        trend=None,
                        seasonal=None,
                        initialization_method="estimated",
                    ).fit()
                forecast = model.forecast(max_horizon)
            except Exception:
                forecast = None

        # Fallback: series_id / globaler Durchschnitt
        if forecast is None:
            fb = series_fallback.get(series_id, global_fallback)
            forecast = np.full(max_horizon, fb)

        for day_idx in val_days:
            step = int(day_idx - train_max_day) - 1  # 0-indexiert
            if 0 <= step < len(forecast):
                pred_value = forecast[step]
            else:
                pred_value = series_fallback.get(series_id, global_fallback)
            predictions[(series_id, day_idx)] = pred_value

    val_pred["prediction"] = val_pred.apply(
        lambda row: predictions.get((row["series_id"], row["day_idx"]), global_fallback),
        axis=1,
    )

    val_pred["prediction"] = val_pred["prediction"].fillna(global_fallback)
    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred

def holt_winters_exp_forecast(train_df, val_df, target_col, seasonal_periods=7):
    """
    Holt-Winters Forecast pro series_id.

    Modelliert:
    - Level
    - Trend
    - saisonales Muster, z. B. Wochenmuster mit seasonal_periods=7

    Bei Fehlern oder zu wenigen Daten wird auf einfachere Methoden zurückgefallen.
    """

    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    import warnings
    import numpy as np

    val_pred = val_df.copy()

    global_fallback = train_df[target_col].mean()
    series_fallback = train_df.groupby("series_id")[target_col].mean()

    predictions = {}

    for series_id, val_group in val_pred.groupby("series_id"):

        train_group = train_df[
            train_df["series_id"] == series_id
        ].sort_values("day_idx")

        val_days = val_group["day_idx"].sort_values().values

        if train_group.empty:
            for day_idx in val_days:
                predictions[(series_id, day_idx)] = global_fallback
            continue

        y = train_group[target_col].astype(float).values
        n = len(y)

        train_max_day = train_group["day_idx"].max()
        max_horizon = int(val_days.max() - train_max_day)
        max_horizon = max(max_horizon, 1)

        forecast = None

        # Holt-Winters: Trend + Saisonalität
        if n >= 2 * seasonal_periods:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    model = ExponentialSmoothing(
                        y,
                        trend="add",
                        damped_trend=True,
                        seasonal="add",
                        seasonal_periods=seasonal_periods,
                        initialization_method="estimated"
                    ).fit()

                forecast = model.forecast(max_horizon)

            except Exception:
                forecast = None

        # Fallback 1: Holt ohne Saisonalität
        if forecast is None and n >= 2:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    model = ExponentialSmoothing(
                        y,
                        trend="add",
                        damped_trend=True,
                        seasonal=None,
                        initialization_method="estimated"
                    ).fit()

                forecast = model.forecast(max_horizon)

            except Exception:
                forecast = None

        # Fallback 2: SES
        if forecast is None and n >= 1:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    model = ExponentialSmoothing(
                        y,
                        trend=None,
                        seasonal=None,
                        initialization_method="estimated"
                    ).fit()

                forecast = model.forecast(max_horizon)

            except Exception:
                forecast = None

        # Fallback 3: series/global mean
        if forecast is None:
            fb = series_fallback.get(series_id, global_fallback)
            forecast = np.full(max_horizon, fb)

        for day_idx in val_days:
            step = int(day_idx - train_max_day) - 1

            if 0 <= step < len(forecast):
                pred_value = forecast[step]
            else:
                pred_value = series_fallback.get(series_id, global_fallback)

            predictions[(series_id, day_idx)] = pred_value

    val_pred["prediction"] = val_pred.apply(
        lambda row: predictions.get(
            (row["series_id"], row["day_idx"]),
            global_fallback
        ),
        axis=1
    )

    val_pred["prediction"] = val_pred["prediction"].fillna(global_fallback)
    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred
def arima(train_df, val_df, target_col, order=(1, 1, 1)):
    """
    ARIMA Forecast pro series_id.

    Für jede series_id wird ein ARIMA-Modell auf den Trainingsdaten
    gefittet und für die Validierungstage vorhergesagt.

    ARIMA(p,d,q):
    - p: Anzahl vergangener Werte
    - d: Anzahl Differenzierungen gegen Trend
    - q: Anzahl vergangener Fehlerterme

    Fallback-Kette:
    1. ARIMA(order)
    2. Seasonal Naive / letzter Wert der Vorwoche, falls möglich
    3. series_id Durchschnitt
    4. globaler Durchschnitt
    """

    val_pred = val_df.copy()

    global_fallback = train_df[target_col].mean()
    series_fallback = train_df.groupby("series_id")[target_col].mean()

    predictions = {}

    for series_id, val_group in val_pred.groupby("series_id"):

        train_group = train_df[
            train_df["series_id"] == series_id
        ].sort_values("day_idx")

        val_days = val_group["day_idx"].sort_values().values

        if train_group.empty:
            fb = global_fallback
            for day_idx in val_days:
                predictions[(series_id, day_idx)] = fb
            continue

        y = train_group[target_col].astype(float).values
        n = len(y)

        train_max_day = train_group["day_idx"].max()

        max_horizon = int(val_days.max() - train_max_day)
        max_horizon = max(max_horizon, 1)

        forecast = None

        # ARIMA braucht genug Werte
        if n >= 10:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    model = ARIMA(
                        y,
                        order=order
                    )

                    fitted = model.fit()

                    forecast = fitted.forecast(steps=max_horizon)

            except Exception:
                forecast = None

        # Fallback 1: letzter verfügbarer Wert
        if forecast is None:
            if n > 0:
                last_value = y[-1]
                forecast = np.full(max_horizon, last_value)
            else:
                fb = series_fallback.get(series_id, global_fallback)
                forecast = np.full(max_horizon, fb)

        for day_idx in val_days:
            step = int(day_idx - train_max_day) - 1

            if 0 <= step < len(forecast):
                pred_value = forecast[step]
            else:
                pred_value = series_fallback.get(series_id, global_fallback)

            predictions[(series_id, day_idx)] = pred_value

    val_pred["prediction"] = val_pred.apply(
        lambda row: predictions.get(
            (row["series_id"], row["day_idx"]),
            global_fallback
        ),
        axis=1
    )

    val_pred["prediction"] = val_pred["prediction"].fillna(global_fallback)
    val_pred["prediction"] = val_pred["prediction"].clip(lower=0)

    return val_pred


def lightgbm_forecast(train_df, val_df, target_col, random_state=42):
    """
    LightGBM Forecast.

    Trainiert ein globales Modell über alle series_id hinweg.
    Zielvariable ist target_col, z. B.:
    - sale_amount
    - recovered_daily_sales_xgboost
    - recovered_daily_sales_hourly_mean

    Das Modell nutzt Zeit-, Serien- und Kontextfeatures.
    """

    import lightgbm as lgb
    import numpy as np
    import pandas as pd

    train = train_df.copy()
    val_pred = val_df.copy()

    # ------------------------------------------------------------
    # 1. Features erstellen
    # ------------------------------------------------------------

    def add_lgbm_features(df):
        df = df.copy()

        df["weekday"] = pd.to_datetime(df["dt"]).dt.weekday
        df["month"] = pd.to_datetime(df["dt"]).dt.month

        # Lag-/Rolling-Features aus sale_amount bzw. vorhandenen Features
        # Falls target_col schon eigene Recovery-Spalte ist, werden Features
        # trotzdem aus dieser Zielspalte gebaut.
        df = df.sort_values(["series_id", "day_idx"])

        grp = df.groupby("series_id")[target_col]

        df["lag1"] = grp.shift(1)
        df["lag7"] = grp.shift(7)
        df["rolling7"] = grp.shift(1).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
        df["rolling28"] = grp.shift(1).rolling(28, min_periods=1).mean().reset_index(level=0, drop=True)

        return df

    combined = pd.concat([train, val_pred], axis=0).sort_values(["series_id", "day_idx"])

    combined = add_lgbm_features(combined)

    train_feat = combined[combined["day_idx"].isin(train["day_idx"])].copy()
    val_feat = combined[combined["day_idx"].isin(val_pred["day_idx"])].copy()

    # ------------------------------------------------------------
    # 2. Feature-Liste
    # ------------------------------------------------------------

    feature_cols = [
        "series_id",
        "product_id",
        "store_id",
        "city_id",
        "management_group_id",
        "weekday",
        "month",
        "day_idx",
        "discount",
        "holiday_flag",
        "activity_flag",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
        "precpt",
        "lag1",
        "lag7",
        "rolling7",
        "rolling28",
        "psd",
    ]

    feature_cols = [c for c in feature_cols if c in train_feat.columns]

    # ------------------------------------------------------------
    # 3. Missing Values behandeln
    # ------------------------------------------------------------

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)

    X_val = val_feat[feature_cols].fillna(0)

    # ------------------------------------------------------------
    # 4. Modell trainieren
    # ------------------------------------------------------------

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="regression",
        n_jobs=-1,
        random_state=random_state,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    # ------------------------------------------------------------
    # 5. Forecast
    # ------------------------------------------------------------

    val_feat["prediction"] = model.predict(X_val)

    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def xgboost_forecast(train_df, val_df, target_col, random_state=42):
    """
    XGBoost Forecast.

    Globales Modell über alle series_id.
    Forecast der nächsten Tage auf Basis von
    Zeit-, Produkt-, Store- und Wetterfeatures.
    """

    import xgboost as xgb
    import numpy as np
    import pandas as pd

    train = train_df.copy()
    val_pred = val_df.copy()

    # ------------------------------------------------------------
    # 1. Features erstellen
    # ------------------------------------------------------------

    def add_xgb_features(df):
        df = df.copy()

        df["weekday"] = pd.to_datetime(df["dt"]).dt.weekday
        df["month"] = pd.to_datetime(df["dt"]).dt.month

        df = df.sort_values(["series_id", "day_idx"])

        grp = df.groupby("series_id")[target_col]

        df["lag1"] = grp.shift(1)
        df["lag7"] = grp.shift(7)

        df["rolling7"] = (
            grp.shift(1)
            .rolling(7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df["rolling28"] = (
            grp.shift(1)
            .rolling(28, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        return df

    combined = pd.concat([train, val_pred], axis=0)
    combined = combined.sort_values(["series_id", "day_idx"])

    combined = add_xgb_features(combined)

    train_feat = combined[
        combined["day_idx"].isin(train["day_idx"])
    ].copy()

    val_feat = combined[
        combined["day_idx"].isin(val_pred["day_idx"])
    ].copy()

    # ------------------------------------------------------------
    # 2. Features auswählen
    # ------------------------------------------------------------

    feature_cols = [
        "series_id",
        "product_id",
        "store_id",
        "city_id",
        "management_group_id",
        "weekday",
        "month",
        "day_idx",
        "discount",
        "holiday_flag",
        "activity_flag",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
        "precpt",
        "lag1",
        "lag7",
        "rolling7",
        "rolling28",
        "psd",
    ]

    feature_cols = [
        c for c in feature_cols
        if c in train_feat.columns
    ]

    # ------------------------------------------------------------
    # 3. Daten vorbereiten
    # ------------------------------------------------------------

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)

    X_val = val_feat[feature_cols].fillna(0)

    # ------------------------------------------------------------
    # 4. Modell trainieren
    # ------------------------------------------------------------

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    # ------------------------------------------------------------
    # 5. Forecast
    # ------------------------------------------------------------

    val_feat["prediction"] = model.predict(X_val)

    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

def random_forest_forecast(train_df, val_df, target_col, random_state=42):
    """
    Random Forest Forecast.

    Globales Modell über alle series_id.
    Nutzt Zeit-, Produkt-, Store- und Wetterfeatures.
    """

    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd

    train = train_df.copy()
    val_pred = val_df.copy()

    # ------------------------------------------------------------
    # Features erstellen
    # ------------------------------------------------------------

    def add_features(df):
        df = df.copy()

        df["weekday"] = pd.to_datetime(df["dt"]).dt.weekday
        df["month"] = pd.to_datetime(df["dt"]).dt.month

        df = df.sort_values(["series_id", "day_idx"])

        grp = df.groupby("series_id")[target_col]

        df["lag1"] = grp.shift(1)
        df["lag7"] = grp.shift(7)

        df["rolling7"] = (
            grp.shift(1)
            .rolling(7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df["rolling28"] = (
            grp.shift(1)
            .rolling(28, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        return df

    combined = pd.concat([train, val_pred], axis=0)
    combined = combined.sort_values(["series_id", "day_idx"])

    combined = add_features(combined)

    train_feat = combined[
        combined["day_idx"].isin(train["day_idx"])
    ].copy()

    val_feat = combined[
        combined["day_idx"].isin(val_pred["day_idx"])
    ].copy()

    # ------------------------------------------------------------
    # Features auswählen
    # ------------------------------------------------------------

    feature_cols = [
        "series_id",
        "product_id",
        "store_id",
        "city_id",
        "management_group_id",
        "weekday",
        "month",
        "day_idx",
        "discount",
        "holiday_flag",
        "activity_flag",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
        "precpt",
        "lag1",
        "lag7",
        "rolling7",
        "rolling28",
        "psd",
    ]

    feature_cols = [
        c for c in feature_cols
        if c in train_feat.columns
    ]

    # ------------------------------------------------------------
    # Daten vorbereiten
    # ------------------------------------------------------------

    global_fallback = train[target_col].mean()

    X_train = train_feat[feature_cols].fillna(0)
    y_train = train_feat[target_col].fillna(global_fallback)

    X_val = val_feat[feature_cols].fillna(0)

    # ------------------------------------------------------------
    # Modell trainieren
    # ------------------------------------------------------------

    # model = RandomForestRegressor (eine runde ca 58 min)
    #     n_estimators=200,
    #     max_depth=15,
    #     min_samples_leaf=20,
    #     n_jobs=-1,
    #     random_state=random_state,
    #     verbose=1
    # )
    model = RandomForestRegressor(
        n_estimators=50,          # statt 200
        max_depth=12,             # statt 15
        min_samples_leaf=50,      # statt 20
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )

    model.fit(X_train, y_train)

    # ------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------

    val_feat["prediction"] = model.predict(X_val)

    val_feat["prediction"] = val_feat["prediction"].clip(lower=0)

    return val_feat

# TODO 
# Exponential Smoothing (Siehe oben) (Nils)
# DLinear (Nils)
# Transformer (Achtung sehr lange Ladezeit evtl. einfach LSTM verwenden) (Laura)
# CNN (Laura)
# Random Forest (Laura)